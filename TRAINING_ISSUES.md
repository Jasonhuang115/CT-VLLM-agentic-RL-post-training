# 训练踩坑记录

> Qwen2.5-VL-3B + Unsloth FastVisionModel 后训练全流程问题记录

---

## #1 Qwen VL 动态分辨率导致 tokenizer 图像占位符不匹配

**日期**: 2026-07

**现象**: `StopIteration` / tokenizer 报错，图像 token 数量对不上

**根因**: Qwen2.5-VL 的 dynamic resolution 机制会把一张 CT 图像拆成 N 个 `<|vision_start|>...<|vision_end|>` 块。tokenizer 需要 N 个图像路径，但代码只传了 1 个。

**解决**: 添加 `match_images()` 辅助函数，统计 text 中的 `<|vision_start|>` 数量，将图像路径重复到匹配数量：

```python
def match_images(text: str, img_paths: list) -> list:
    n = text.count('<|vision_start|>')
    imgs = list(img_paths)
    while len(imgs) < n:
        imgs = imgs + img_paths
    return imgs[:n]
```

**影响范围**: Stage 1/2/3/4a/4b 全部训练脚本

---

## #2 FlashAttention2 ImportError

**日期**: 2026-07

**现象**: `ImportError: FlashAttention2 has been toggled on, but it cannot be used`

**根因**: AutoDL 实例上 flash-attn 未安装，但模型配置默认开启了 FlashAttention2

**解决**: `attn_implementation="sdpa"` 替代 FlashAttention2

---

## #3 Stage 1 SFT Loss 异常 (16877 / 2095)

**日期**: 2026-07-07

**现象**: SFT 训练 loss 显示 16877（第一次）或 2095（修复后），远高于正常的 2~5

**根因 (两重 bug)**:

1. **Loss 计算范围错误**: 原始代码对全部 token（system prompt + 用户指令 + 图像 patch token + assistant 回复）计算 CE loss。图像 token 占 2000~4000 个位置，不应参与 loss 计算。SFT 只应该对 assistant 回复部分计算 loss。

2. **FastVisionModel 内部使用 sum reduction**: 当 `model(**enc, labels=labels)` 传入 labels 时，Qwen2.5-VL 内部使用 `CrossEntropyLoss(reduction='sum')`（求和而非求平均）。修复后仍显示 2095，因为 `out.loss` 不是 None，代码走了模型的 sum reduction 分支。

**修复（最终版本 — 经 3 次迭代）**:

```python
# 1. 用 prompt-only encoding 找到 response 边界
prompt_txt = tok.apply_chat_template(
    [m for m in msgs if m["role"] != "assistant"],
    tokenize=False, add_generation_prompt=True)
prompt_enc = tok(text=[prompt_txt], images=images_for_tok, return_tensors="pt")
prompt_len = prompt_enc.input_ids.shape[1]

# 2. Mask 掉 prompt 部分（包括图像 token）
labels = enc["input_ids"].clone()
labels[:, :prompt_len] = -100

# 3. 传 labels 给模型 → fused kernel 高效算 loss（不物化 logits）
#    然后手动把 sum loss 转成 mean loss
out = model(**enc, labels=labels)          # 内存高效
loss_sum = out.loss                         # sum reduction
n_tokens = (labels[:, 1:] != -100).sum()   # 非 mask token 数
loss = loss_sum / max(n_tokens, 1)          # → mean ≈ 2~5
```

**为什么这么做**:
- ✅ 传 `labels` → fused loss kernel，不物化 `[L, 152064]` 的 logits，省 5GB
- ✅ `labels` mask → 只算 response token，不被图像 token 干扰
- ✅ 手动 `/ n_tokens` → 把 sum 转成 mean，loss 在 2~5 可读范围

**迭代记录**（3 次修复才到位）:

| 版本 | 方法 | 问题 |
|---|---|---|
| 原始 | 无 mask + out.loss sum | 16877 |
| #1 | mask + out.loss sum | 2095（sum，不是 mean）|
| #2 | mask + 手动 CE（不传 labels）| CUDA OOM（物化 logits）|
| #3 ✅ | mask + out.loss sum / n_tokens | **正确 mean + 省显存** |

**关键教训**:
- VLM 训练必须 response-only loss masking
- Qwen2.5-VL / Unsloth 内部 loss 是 sum reduction，需手动转 mean
- `model(**enc)` 不传 labels 会物化完整 logits `[L, 152064]` → 显存灾难
- `model(**enc, labels=labels)` 用 fused kernel → 内存高效但返回 sum loss
- **正确姿势**: 传 labels + 手动 sum/n_tokens → mean

---

## #4 zip 炸弹检测导致大文件解压失败

**日期**: 2026-07

**现象**: AutoDL 上 `unzip subset3.zip` 失败

**根因**: Linux unzip 对超过一定大小的 zip 文件触发炸弹检测，拒绝解压

**解决**: 使用 `7z x subset3.zip` 替代 unzip

---

## #5 subset7 孤立 .raw 文件（缺 .mhd 头)

**日期**: 2026-07

**现象**: 目录里有 526 个 .raw 但只有 439 个 .mhd，87 个 .raw 孤立的无法被 SimpleITK 读取

**根因**: subset7 从 Mac 手动上传时只上传了 .raw，漏了 .mhd 头文件

**解决**: 
1. 写 `scripts/check_orphan_raw.py` 脚本定位孤立的 .raw
2. 用 `rsync *.mhd` 从 Mac 补传 subset7 的 .mhd 文件
3. 发现部分 .raw 本身也已损坏（unzip 截断），需 rsync 完整覆盖 subset7 全部 .raw

---

## #6 Stage 1 SFT CUDA OOM — 手动 CE 导致 logits 显存爆炸

**日期**: 2026-07-07

**现象**: `torch.cuda.OutOfMemoryError` 在 stage1_sft.py 启动后 OOM

**根因**: Issue #3 的修复引入了新问题。之前的修复为了绕过模型内部 sum reduction，调用 `model(**enc)`（不传 labels）来获取 logits 然后手动算 CE。但这会物化完整 logits 张量 `[1, L, 152064]`。

对于 Qwen2.5-VL 动态分辨率，L 可达 3000~8000：
- fp32: 8000 × 152064 × 4 bytes ≈ **4.9 GB**（仅 logits，不含模型和优化器）
- 加上 4-bit 模型 ~4 GB + 优化器状态 + 梯度 → 24GB 显存直接爆

**为什么传 labels 时不会 OOM**: 模型内部计算 loss 时不需要物化完整 logits——可以用 fused kernel 分块计算，logits 不离开 GPU 临时缓存。

**修复**: 回到传 labels 的方式，但手动把 sum loss 转换成 mean loss：

```python
# 传 labels 给模型（内存高效的内部 loss 计算）
out = model(**enc, labels=labels)
loss_sum = out.loss  # 模型内部默认 sum reduction
n_tokens = (labels[:, 1:] != -100).sum().item()  # 非 mask token 数
loss = loss_sum / max(n_tokens, 1)  # 手动转 mean
```

**关键教训**:
- `model(**enc)` 不使用 labels 会物化完整 logits，内存灾难
- `model(**enc, labels=labels)` 使用 fused loss kernel，无 logits 物化
- 正确做法是用模型的 sum loss ÷ token 数，而非手动算 CE
- VLM + Qwen 大词表 (152K) + 长序列 = 绝对不能随意物化 logits

---

## #7 UnslothVisionTrainer 导入失败

**日期**: 2026-07

**现象**: `ImportError: cannot import name 'UnslothVisionTrainer' from 'unsloth'`

**根因**: Mac 上的 Unsloth 版本比 AutoDL 高，`UnslothVisionTrainer` 是较新版本才有的类。Mac 写脚本时用了新 API，传到 AutoDL 后旧版 Unsloth 找不到。

**解决**: 放弃 TRL Trainer 封装，改用手动训练循环（FastVisionModel 直接 forward + backward）

---

## #8 Stage 3 GRPO composite_reward 导入错误

**日期**: 2026-07

**现象**: `ImportError: cannot import name 'composite_score'`

**根因**: reward 函数实际叫 `composite_reward`，参数格式为 `(completion, ground_truth_dict, lang, reward_weights)`，其中 `ground_truth_dict` 需包含 `long_diameter_mm`, `short_diameter_mm`, `malignancy_level`, `characteristics` 字段

**解决**: 修正 import 名称 + 正确的 ground_truth dict 格式

---

## #9 subset7 .raw 文件损坏（unzip 截断）

**日期**: 2026-07

**现象**: `mhd_to_png` 报错 `M_ReadElementsData: data not read completely`，大量 subset7 的 .raw 无法转 PNG

**根因**: subset7 zip 在 AutoDL 上 unzip 解压时中途截断，部分 .raw 文件不完整。Mac 上原始 zip 解压正常，但上传到 AutoDL 后用了有问题的 unzip 导致

**解决**: 从 Mac 用 `rsync` 直接覆盖 subset7 全部 .raw 文件（跳过 unzip 步骤）
```bash
rsync -avP -e "ssh -p 25017" \
  /Users/dp/Downloads/LUNA16/subset7/subset7/*.raw \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/data/LUNA16/images/
```

---

## #10 Stage 2 SimPO 图像被丢弃（v2 无条件模式）

**日期**: 2026-07

**现象**: Stage 2 训练 acc=0.50（等于随机），模型没学到任何图像信息

**根因**: 与 Issue #1 同源——prompt/chosen/rejected 三个文本在 tokenize 时只传了 1 个图像路径，Qwen dynamic-res 需要 N 个。所有 `vlm_tok()` 调用都缺少 `match_images()` 处理

**解决**: 所有 tokenizer 调用点统一使用 `match_images(text, image_paths)` 替代原始 `image_paths`

**验证**: 30 样本测试 acc=0.7778（v3 修复后）

---

## #11 TRL SFTConfig pickle 序列化失败 → wandb 误报 Failed

**日期**: 2026-07

**现象**: Stage 1 checkpoint 保存时报错 `_pickle.PicklingError: Can't pickle <class 'trl.trainer.sft_config.SFTConfig'>`，wandb 面板显示 "Failed"

**根因**: TRL 的 SFTConfig 对象在某些 Unsloth/transformers 版本组合下无法被 pickle 序列化。`trainer.save_model()` 内部会尝试序列化 trainer state → 失败 → Python 进程异常退出 → wandb 看到非零 exit code → 标记 Failed。但模型权重实际已保存成功。

**解决**: try/except 包裹 `trainer.save_model()`，fallback 到 `model.save_pretrained() + tokenizer.save_pretrained()`
```python
try:
    trainer.save_model(output_dir)
except (TypeError, pickle.PicklingError):
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
```

**教训**: wandb "Failed" 不一定等于训练失败——可能只是保存阶段的序列化问题

---

## #12 Unsloth VLM tokenizer 调用格式不兼容

**日期**: 2026-07

**现象**: `TypeError: only a single or a list of entries is supported but got type=<class 'dict'>`

**根因**: Unsloth 2026.6.9 版本 patch 了 Qwen2.5-VL processor 的 `__call__`，所有 tokenizer 调用都路由到 image_processor。VLM tokenizer 只接受 `tok(text=[text], images=[path])` 格式（两个独立参数），不能传标准的 messages dict。

**解决**: 
```python
# ❌ 错误 — messages dict 格式
tok.apply_chat_template(messages, ...)
# ✅ 正确 — 先手动 apply_chat_template 得到 text，再分别传 text 和 images
text = tok.apply_chat_template(messages, tokenize=False, ...)
enc = tok(text=[text], images=[image_path], return_tensors="pt")
```

---

## #13 lidc_match pylidc/LUNA16 坐标系不匹配

**日期**: 2026-07

**现象**: lidc_match 匹配率 0/70 → 31/60 → 0/68 → 47/67 → 最终 62/62

**根因 (三层错误)**:
1. **坐标空间混淆**: pylidc `ann.centroid` 返回 **voxel 索引**（不是 world mm），但 LUNA16 coordX/Y/Z 是 **world 坐标 (mm)**
2. **整数截断**: `TransformPhysicalPointToIndex()` 取整丢失精度，需用 `TransformPhysicalPointToContinuousIndex()`
3. **轴顺序不确定**: pylidc 和 LUNA16 的 X/Y 轴可能互换，必须比较全部 4 种组合

**最终方案**:
```python
ct_image = sitk.ReadImage(mhd_path)
voxel_xyz = ct_image.TransformPhysicalPointToContinuousIndex((lx, ly, lz))
voxel_yxz = ct_image.TransformPhysicalPointToContinuousIndex((ly, lx, lz))
# 4 种组合取最小距离
d1 = np.linalg.norm(voxel_xyz - [px, py, pz])
d2 = np.linalg.norm(voxel_xyz - [py, px, pz])
d3 = np.linalg.norm(voxel_yxz - [px, py, pz])
d4 = np.linalg.norm(voxel_yxz - [py, px, pz])
d = min(d1, d2, d3, d4)
threshold = max(15.0, 5.0 * diameter / min(spacing))
```

---

## #14 Zenodo CDN zip 下载损坏 + scp 断连

**日期**: 2026-07

**现象**: Mac 上所有 9 个 subset zip 文件 `End-of-central-directory signature not found`；AutoDL unzip 失败；subset5 只有 4.45G（实际应 ~6G）

**根因 (两重)**:
1. Zenodo CDN 重定向 + `curl -C -` 断点续传时，CDN 节点切换导致 central directory 损坏
2. scp 传输大文件中途断连无法续传

**解决**:
- Mac 端: `zip -FF subsetX.zip --out subsetX_fixed.zip` 修复 central directory
- 上传: 用 `rsync -avP -e "ssh -p 25017"` 替代 scp（支持断点续传）
- AutoDL 解压: 大文件用 `7z x` 替代 `unzip`（避免 zip 炸弹检测）

**教训**: 
- Zenodo/CDN 下载大文件不要用 `-C -`（断点续传），直接重新下载
- 远程传输永远用 rsync，不要用 scp
- Mac zip 和 Linux unzip 有兼容性差异，zip -FF 可修复大部分问题

---

## #15 SFT loss=6.4 停滞 — 模型泛泛输出"背教科书"

**日期**: 2026-07-07

**现象**: Stage 1 SFT 训完 3 epoch, loss=6.4 (ppl≈600)。生成报告示例：
```
"图像左侧的肺部区域存在一个大小约2cm的圆形结节"  ← 幻觉(GT=10mm)
"直径小于3厘米...通常被认为是良性的可能性更大"    ← 教科书式泛泛描述
```
完全不提任何 LIDC 特征（密度类型/边界评分/毛刺/钙化/Lung-RADS）。

**根因 (专家诊断，三重致命缺陷)**:

1. **数据管线方向性错误（最致命）**: DeepSeek API 基于 LIDC 文本特征生成报告，**未看 CT 图像**。但 assistant 报告里包含"右上叶后段/边界欠清/实性/无钙化"等信息——这些在单张 2D 512×512 灰度 PNG 上根本不一定看得出来。模型被要求学一个 **图像 X → 文本 Y** 的映射，但 **Y 的信息量 > X 能承载的信息量**。模型只能选择"背教科书泛泛输出"来最大化似然。

2. **Vision encoder 全冻结**: ViT-G/14 在 LAION 自然图像上预训练，从未见过 CT 灰度图。`finetune_vision_layers=False` 冻结全部视觉层 → 视觉信号根本流不进 LLM。特别是 vision-language projector/connector 层的冻结，等于让 LLM"瞎猜"。

3. **中文 token 稀疏放大了 ppl**: Qwen 词表 152K 中医学中文覆盖率低，"毛刺征"可能被拆成 3-4 个字节 token → 每个 token 概率都被稀释 → perplexity 偏高。

**验证方法**:
- 中英对照: 同一条数据的英文版 loss 是否显著低于中文 → 确认 token 稀疏贡献
- 打印 label tensor: 确认 response mask 正确，非 -100 token 数 = assistant 实际 token 数

**修复计划 (见 PLAN.md)**:
| 优先级 | 措施 | 说明 |
|---|---|---|
| **P1** | 结构化 prompt | user prompt 注入坐标+直径，让模型有锚点 |
| **P2** | 解冻 vision projector + ViT 最后2层 | 分组 LR (connector正常, ViT 1/10) |
| **P4** | ReST 替代 GRPO | SFT 收敛后再自蒸馏，不先做 RL |

**核心教训**:
- **不要让没看图的 LLM 生成报告去教要看图的 VLM** — 这是概念级的方向性错误
- Freeze vision encoder + CT domain gap = 视觉信号断裂
- 3B 不是问题的根因，数据管线 + vision 冻结才是
- SFT 没收敛前做 GRPO/RL = 奖励欺骗

---

## #16 (P0 待执行) Response mask 正确性验证 + 中文 token 稀疏排查

**日期**: 2026-07-07 (计划中)

**待验证**:
- [ ] 打印一条样本的 label tensor，确认 `labels != -100` 的 token 数 = assistant 回复的实际 token 数
- [ ] 确认 `<|vision_start|>`, `<|image_pad|>`, `<|vision_end|>` 等特殊 token 都被 mask
- [ ] 中英对照 loss: 同一条数据的中文版 vs 英文版 loss 差异

---

## 已预见但尚未遇到
- [ ] Stage 3 GRPO reward 方差过小导致梯度消失
- [ ] Stage 4a/4b Agentic 训练轨迹格式不匹配

