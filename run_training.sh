#!/bin/bash

echo "=== Qwen2.5-VL LoRA微调训练 ==="

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到NVIDIA GPU"
    exit 1
fi

# 显示GPU信息
echo "GPU信息:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# 检查显存
gpu_memory=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
if [ "$gpu_memory" -lt 16000 ]; then
    echo "警告: 可用显存不足16GB (当前: ${gpu_memory}MB)"
    echo "建议停止其他GPU进程或使用更小的模型"
fi

# 安装依赖
echo "安装依赖包..."
pip install -r requirements.txt

# 数据处理
echo "开始数据处理..."
python process_data.py

# 检查数据是否处理成功
if [ ! -f "data/train.json" ]; then
    echo "错误: 数据处理失败，未找到训练数据文件"
    exit 1
fi

echo "数据处理完成"

# 开始训练
echo "开始LoRA微调训练..."
python qwen25vl_lora_train.py

echo "训练完成！" 