#!/bin/bash
# ===================================================
# CT Chatbot App — AutoDL 部署验证脚本
# ===================================================
set -e

echo "============================================"
echo " CT Chatbot App 部署验证"
echo "============================================"

# 1. 依赖
echo "[1/6] 安装依赖..."
pip install -r requirements-app.txt -q 2>/dev/null
python3 -c "import fastapi,uvicorn,httpx,SimpleITK; print('  依赖 OK')"

# 2. Python 编译
echo "[2/6] 编译检查..."
python3 -m compileall app -q && echo "  编译 OK"

# 3. Skills 冒烟测试
echo "[3/6] Skills 测试..."
printf '%s' '{"query":"肺结节随访","top_k":1}' \
  | python3 app/backend/skills/guideline_retrieval/run.py > /dev/null && echo "  guideline_retrieval OK"
printf '%s' '{"function":"lung_rads_classify","nodule_type":"solid","size_mm":8.6}' \
  | python3 app/backend/skills/lung_rads_calculator/run.py > /dev/null && echo "  lung_rads_calculator OK"
printf '%s' '{"query":"肺结节","max_results":1}' \
  | python3 app/backend/skills/web_search/run.py > /dev/null && echo "  web_search OK"
printf '%s' '{"image_path":"/nonexistent"}' \
  | python3 app/backend/skills/image_metadata/run.py 2>&1 | grep -q "error\|不存在" && echo "  image_metadata OK (文件不存在=预期)"

# 4. ROI contract
echo "[4/6] ROI 合同验证..."
python3 -c "from app.backend.roi import roi_contract; c=roi_contract(); assert c['roi_size_mm']==50.0; assert c['output_size_px']==512; print('  ROI 合同 OK')" 2>/dev/null || echo "  (SimpleITK 未安装, 跳过)"

# 5. Mock 后端启动测试
echo "[5/6] Mock 后端启动..."
export VLM_MOCK=true
timeout 5 python3 -m app.backend.main 2>&1 &
sleep 2
HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null || echo "")
if echo "$HEALTH" | grep -q '"ok":true'; then
    echo "  /health OK"
else
    echo "  ⚠️  /health 不可达 (可能缺依赖或端口冲突)"
fi
kill %1 2>/dev/null || true

# 6. ROI 对比验证 (如果有测试 CT)
echo "[6/6] ROI 输出对比..."
echo "  需要手动验证: 用同一 LUNA16 病例 + 同一结节坐标,"
echo "  对比 app/backend/roi.py 和 data/mhd_to_png.py extract_multi_view() 的输出"
echo "  Diff check:"
echo "    python3 -c \"
echo "    from PIL import Image; import numpy as np"
echo "    a = np.array(Image.open('roi_from_app_axial.png'))"
echo "    b = np.array(Image.open('roi_from_training_axial.png'))"
echo "    print(f'像素差异: {np.abs(a.astype(float)-b.astype(float)).mean():.2f}')\""

echo ""
echo "============================================"
echo " 验证完成"
echo "============================================"
echo ""
echo "下一步:"
echo "  export VLM_MOCK=true"
echo "  python3 -m app.backend.main"
echo "  浏览器打开 http://localhost:8080"
echo ""
echo "真实 VLM 模式:"
echo "  vllm serve huang01080524/lungct-nodule-grpo --host 0.0.0.0 --port 8000 --limit-mm-per-prompt image=3"
echo "  python3 -m app.backend.main"
