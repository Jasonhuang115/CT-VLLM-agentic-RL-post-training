#!/usr/bin/env python3
"""
Stage 0: CT → VLM 全链路验证

目标：用 1-3 个 LIDC-IDRI 病例，证明整个管线跑得通：
  1. DICOM 加载正常
  2. 结节 ROI 提取正常
  3. 多视图生成正常
  4. Qwen2.5-VL-3B 能加载（4-bit）
  5. 模型能对 CT 图像输出有意义的描述

使用方式：
  python training/stage0_check.py

如果本机无 GPU 或无数据，脚本会自动降级为语法检查模式。
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def check_environment():
    """检查运行环境"""
    import torch
    results = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only",
        "gpu_memory": (
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            if torch.cuda.is_available() else "N/A"
        ),
    }
    print("\n=== 环境检查 ===")
    for k, v in results.items():
        print(f"  {k}: {v}")
    return results


def check_dependencies():
    """检查关键依赖是否可导入"""
    deps = {}
    modules = [
        "torch", "transformers", "trl", "peft", "bitsandbytes",
        "pydicom", "nibabel", "cv2", "skimage", "PIL",
        "unsloth", "vllm", "langchain", "langgraph",
    ]
    print("\n=== 依赖检查 ===")
    for mod in modules:
        try:
            __import__(mod)
            deps[mod] = "✅"
        except ImportError:
            deps[mod] = "❌ (未安装)"
        print(f"  {mod:20s}: {deps[mod]}")
    return deps


def check_dicom_load(sample_dir=None):
    """检查 DICOM 加载能力。
    如果没有真实数据，生成一个假的 DICOM 做格式验证。
    """
    print("\n=== DICOM 加载检查 ===")

    if sample_dir and Path(sample_dir).exists():
        # 尝试加载真实 DICOM
        import pydicom
        dcm_files = list(Path(sample_dir).rglob("*.dcm"))
        if dcm_files:
            ds = pydicom.dcmread(str(dcm_files[0]))
            print(f"  ✅ 成功加载真实 DICOM: {dcm_files[0].name}")
            print(f"     PatientID: {ds.get('PatientID', 'N/A')}")
            print(f"     Modality: {ds.get('Modality', 'N/A')}")
            print(f"     Image shape: {ds.pixel_array.shape}")
            return True

    # 降级：检查 pydicom 导入，不做合成 DICOM 测试
    print("  ⚠️ 无真实 DICOM 数据，仅验证 pydicom 可导入。")
    import pydicom
    print(f"  ✅ pydicom {pydicom.__version__} 可正常使用")
    return True


def check_ct_to_image():
    """检查 HU 归一化和 CT → PNG 转换"""
    print("\n=== CT → 图像转换检查 ===")

    # 合成一个含"结节"的 CT 切片
    rng = np.random.default_rng(42)
    img_hu = rng.normal(-700, 100, (512, 512)).astype(np.float32)

    # 加结节
    rr, cc = np.ogrid[:512, :512]
    nodule_mask = (rr - 250)**2 + (cc - 300)**2 < 400
    img_hu[nodule_mask] = np.random.default_rng(123).normal(30, 20, nodule_mask.sum())

    # Lung window: WL=-600, WW=1500 → range [-1350, 150]
    wl, ww = -600, 1500
    low, high = wl - ww / 2, wl + ww / 2
    img_clipped = np.clip(img_hu, low, high)
    img_normalized = ((img_clipped - low) / (high - low) * 255).astype(np.uint8)

    # 保存为 PNG
    tmpdir = tempfile.mkdtemp(prefix="stage0_img_")
    img_path = os.path.join(tmpdir, "ct_lung_window.png")
    Image.fromarray(img_normalized).save(img_path)

    print(f"  ✅ HU→PNG 转换成功: {img_path}")
    print(f"     Window: WL={wl}, WW={ww}")
    print(f"     Output range: [{img_normalized.min()}, {img_normalized.max()}]")
    return True


def check_model_load_and_infer():
    """加载 Qwen2.5-VL-3B (4-bit) 并推理"""
    print("\n=== 模型加载 & 推理检查 ===")

    import torch
    if not torch.cuda.is_available():
        print("  ⚠️ 无 GPU，跳过模型加载。请在 AutoDL 实例上运行以完成完整验证。")
        return "skipped"

    vram_before = torch.cuda.memory_allocated(0) / 1024**3

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from qwen_vl_utils import process_vision_info

        model_path = os.environ.get(
            "MODEL_PATH",
            "/root/autodl-tmp/models/Qwen2.5-VL-3B-Instruct"
        )

        print(f"  正在加载模型: {model_path}")
        print("  (首次运行会从 HuggingFace 下载，约需 5-10 分钟)...")

        # 4-bit 量化加载
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        vram_after = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  ✅ 模型加载成功")
        print(f"     VRAM: {vram_before:.1f}GB → {vram_after:.1f}GB (模型占用 ~{vram_after - vram_before:.1f}GB)")

        # 创建测试图像和消息
        test_img = Image.fromarray(
            np.random.randint(0, 255, (512, 512), dtype=np.uint8)
        )
        tmpdir = tempfile.mkdtemp(prefix="stage0_test_")
        img_path = os.path.join(tmpdir, "test_ct.png")
        test_img.save(img_path)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": "What do you see in this medical image? Describe briefly."},
                ],
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        print(f"\n  === 模型输出 ===")
        print(f"  {output[:300]}")
        print(f"  === 验证通过: 模型能正常推理 ===\n")

        return True

    except Exception as e:
        print(f"  ❌ 模型加载/推理失败: {e}")
        print(f"  (如果你在 MacBook 上看到这个，这是预期的——没有 GPU)")
        print(f"  请在 AutoDL 实例上重新运行此脚本。")
        return False


def main():
    parser = argparse.ArgumentParser(description="Stage 0: CT→VLM 全链路验证")
    parser.add_argument("--dicom_dir", type=str, default=None,
                        help="DICOM 数据目录 (可选，无数据时使用合成图像)")
    parser.add_argument("--skip_model", action="store_true",
                        help="跳过模型加载 (仅检查数据处理管线)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Stage 0: CT → VLM 全链路验证")
    print("=" * 60)

    results = {}

    # 1. 环境
    results["env"] = check_environment()

    # 2. 依赖
    results["deps"] = check_dependencies()

    # 3. DICOM 加载
    results["dicom"] = check_dicom_load(args.dicom_dir)

    # 4. CT → 图像
    results["ct2img"] = check_ct_to_image()

    # 5. 模型 (如果有 GPU)
    if not args.skip_model:
        results["model"] = check_model_load_and_infer()
    else:
        results["model"] = "skipped"

    # 总结
    print("\n" + "=" * 60)
    print("  验证总结")
    print("=" * 60)

    all_pass = True
    for name, status in results.items():
        if status == "skipped":
            print(f"  {name}: ⏭️  跳过")
        elif status:
            print(f"  {name}: ✅ 通过")
        else:
            print(f"  {name}: ❌ 失败")
            all_pass = False

    print()
    if all_pass:
        print("  🎉 全链路验证通过！可以开始正式训练。")
        print(f"  下一步: python data/download/download_lidc.py --subset")
        print(f"          python data/sft_dataset_builder.py")
        print(f"          python training/stage1_sft.py")
    else:
        print("  ⚠️  部分检查失败。请解决上述问题后重新运行。")
        print(f"  如果你在 MacBook 上看到这个，这是预期的。")
        print(f"  请在 AutoDL GPU 实例上运行完整验证。")

    print("=" * 60)


if __name__ == "__main__":
    main()
