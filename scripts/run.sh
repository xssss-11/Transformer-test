#!/bin/bash

# ==================================================
# Transformer 模型一键运行脚本
# ==================================================
# 参数说明：
#   - 序列长度 (seq_len): 128
#   - 批次大小 (batch_size): 32
#   - 训练轮数 (n_epochs): 50
#   - 学习率 (lr): 1e-4
#   - 模型维度 (d_model): 128
#   - 注意力头数 (num_heads): 8
#   - 编码器层数 (num_layers): 4
#   - 前馈网络维度 (d_ff): 512
#   - Dropout率: 0.1
# ==================================================

# 设置错误处理
set -e

echo "=========================================="
echo "Transformer 模型训练与测试脚本"
echo "=========================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3，请先安装Python"
    exit 1
fi

# 检查必要的Python包
echo "检查Python依赖包..."
python3 -c "import torch, torch.nn, requests, os, matplotlib" 2>/dev/null || {
    echo "安装必要的Python包..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install requests matplotlib numpy
}

# 创建必要的目录结构
echo "创建目录结构..."
mkdir -p Transformer/data
mkdir -p Transformer/results

# 检查数据文件是否存在
if [ ! -f "Transformer/data/tiny_shakespeare.txt" ]; then
    echo "下载训练数据..."
    cd Transformer/data
    wget -q https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tiny-shakespeare/tiny_shakespeare.txt || {
        echo "使用备用下载源..."
        curl -s -o tiny_shakespeare.txt "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tiny-shakespeare/tiny_shakespeare.txt"
    }
    cd ../..
    
    if [ ! -f "Transformer/data/tiny_shakespeare.txt" ]; then
        echo "错误: 无法下载训练数据"
        exit 1
    fi
fi

echo "数据文件准备就绪"

# 检查模型文件是否存在
if [ ! -f "model.py" ]; then
    echo "错误: 未找到 model.py 文件"
    echo "请确保 Transformer 模型定义文件存在"
    exit 1
fi

# 训练模型
echo "=========================================="
echo "开始训练 Transformer 模型"
echo "=========================================="
echo "超参数配置:"
echo "  - 序列长度: 128"
echo "  - 批次大小: 32" 
echo "  - 训练轮数: 50"
echo "  - 学习率: 1e-4"
echo "  - 模型维度: 128"
echo "  - 注意力头数: 8"
echo "  - 编码器层数: 4"
echo "  - 前馈网络维度: 512"
echo "  - Dropout率: 0.1"
echo "=========================================="

# 运行训练脚本
python3 train.py

# 检查训练是否成功
if [ $? -eq 0 ] && [ -f "Transformer/results/transformer_model_final.pth" ]; then
    echo "=========================================="
    echo "模型训练完成!"
    echo "=========================================="
else
    echo "警告: 训练过程可能存在问题"
    if [ ! -f "Transformer/results/transformer_model_final.pth" ]; then
        echo "错误: 未生成模型文件"
        exit 1
    fi
fi

# 测试模型
echo "=========================================="
echo "开始测试模型生成能力"
echo "=========================================="

# 运行测试脚本
python3 test.py

# 显示结果文件
echo "=========================================="
echo "生成的文件:"
echo "=========================================="
ls -la Transformer/results/

# 显示训练曲线（如果生成了）
if [ -f "Transformer/results/loss_curve.png" ]; then
    echo "=========================================="
    echo "训练损失曲线已保存: Transformer/results/loss_curve.png"
    echo "=========================================="
    
    # 尝试显示图片（如果在图形界面中运行）
    if [ -n "$DISPLAY" ] && command -v xdg-open &> /dev/null; then
        echo "是否打开损失曲线图片? [y/N]"
        read -r response
        case "$response" in
            [yY][eE][sS]|[yY])
                xdg-open "Transformer/results/loss_curve.png" 2>/dev/null || echo "无法自动打开图片，请手动查看"
                ;;
        esac
    fi
fi

echo "=========================================="
echo "所有任务完成!"
echo "=========================================="
echo "输出文件:"
echo "  - 模型文件: Transformer/results/transformer_model_final.pth"
echo "  - 最佳模型: Transformer/results/best_model.pth" 
echo "  - 训练曲线: Transformer/results/loss_curve.png"
echo "  - 检查点: Transformer/results/checkpoint_epoch*.pth"
echo "=========================================="

# 快速验证模型文件
echo "快速验证模型文件..."
python3 -c "
import torch
try:
    model = torch.load('Transformer/results/transformer_model_final.pth', map_location='cpu')
    print('✓ 模型文件验证成功')
    if isinstance(model, dict):
        print(f'  - 检查点包含键: {list(model.keys())}')
    else:
        print(f'  - 模型参数数量: {sum(p.numel() for p in model.parameters() if hasattr(model, \"parameters\"))}')
except Exception as e:
    print(f'✗ 模型文件验证失败: {e}')
"

echo "运行完成!"