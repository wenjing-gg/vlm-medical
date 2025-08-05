# Qwen2.5-VL LoRA微调框架

本项目用于对Qwen2.5-VL-3B-Instruct模型进行LoRA微调，使用医学多模态数据集。支持Docker容器化部署。

## 项目结构

```
qwen2.5vl_lora_finetune/
├── data/                    # 处理后的数据
├── output/                  # 训练输出
├── logs/                    # 日志文件
├── requirements.txt         # 依赖包
├── process_data.py         # 数据处理脚本
├── qwen25vl_lora_train.py # Qwen2.5-VL LoRA训练脚本
├── run_training.sh        # 启动脚本
├── config.yaml            # 配置文件
├── Dockerfile             # Docker镜像构建文件
├── docker-compose.yml     # Docker Compose配置
├── .dockerignore          # Docker忽略文件
├── .gitignore             # Git忽略文件
└── README.md              # 说明文档
```

## 环境要求

- Python 3.8+
- CUDA 11.8+
- 至少16GB GPU内存
- Docker & Docker Compose (可选)

## 安装依赖

### 本地安装
```bash
pip install -r requirements.txt
```

### Docker安装
```bash
# 构建镜像
docker build -t qwen25vl-lora-training .

# 或使用docker-compose
docker-compose build
```

## 使用方法

### 1. 数据处理

```bash
python process_data.py
```

这将下载并处理FreedomIntelligence/Medical_Multimodal_Evaluation_Data数据集。

### 2. 开始训练

#### 本地训练
```bash
python qwen25vl_lora_train.py
```

或者使用启动脚本：
```bash
bash run_training.sh
```

#### Docker训练
```bash
# 使用docker run
docker run --gpus all -v $(pwd)/data:/app/data -v $(pwd)/output:/app/output qwen25vl-lora-training

# 或使用docker-compose
docker-compose up
```

## Docker部署

### 构建镜像
```bash
docker build -t qwen25vl-lora-training .
```

### 运行容器
```bash
# 使用docker-compose（推荐）
docker-compose up -d

# 或直接使用docker run
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 8888:8888 \
  qwen25vl-lora-training
```

### 查看日志
```bash
docker-compose logs -f
```

## 配置说明

主要配置参数在`config.yaml`中：

- **模型配置**: 模型名称、最大长度、图像尺寸
- **数据配置**: 数据路径、批次大小、工作进程数
- **训练配置**: 训练轮数、学习率、权重衰减等
- **LoRA配置**: LoRA参数设置
- **优化配置**: 精度、量化等优化选项

## 训练参数

- **LoRA Rank**: 16
- **LoRA Alpha**: 32
- **学习率**: 2e-4
- **批次大小**: 1
- **训练轮数**: 3
- **量化**: 4bit量化

## 输出文件

训练完成后，模型将保存在`output/`目录中：

- `output/epoch_X/`: 每个epoch的检查点
- `output/best_model/`: 最佳模型

## 注意事项

1. 确保有足够的GPU内存
2. 首次运行会下载模型和数据集，需要较长时间
3. 建议使用tmux会话运行长时间训练
4. Docker版本需要安装nvidia-docker2

## 故障排除

1. **内存不足**: 减小batch_size或使用gradient_accumulation_steps
2. **下载失败**: 检查网络连接，可能需要配置代理
3. **CUDA错误**: 检查CUDA版本兼容性
4. **Docker GPU问题**: 确保安装了nvidia-docker2并正确配置

## 许可证

MIT License 