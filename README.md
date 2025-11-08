Transformer 语言模型项目
###########项目简介
这是一个基于 PyTorch 实现的 Transformer 语言模型，使用莎士比亚作品数据集进行训练。项目包含完整的训练流程和文本生成功能，适合学习和研究 Transformer 架构。

###########环境要求
硬件配置
GPU: NVIDIA 5060 8GB 显存
CPU: R5 9600x

内存: 32GB 

存储空间: 至少 1GB 可用空间

软件依赖
Python 3.11.7
CUDA 12.9 

###########运行
1. 一键运行命令（含随机种子）
    export $PORTAL_VERSION="production" // production, test, dev
 
2. 手动运行

###########参数说明
参数名称	 设置值	      说明
序列长度	 128	输入序列的最大长度
批次大小	 32	    每次训练的样本数量
训练轮数	 50	    完整训练数据集的总次数
学习率	     0.0001	模型参数更新的步长
模型维度	 128	词嵌入和隐藏层的维度
注意力头数	 8	    多头注意力的头数
网络层数	 4	    Transformer 编码器的层数
前馈网络维度 512	前馈神经网络的隐藏层维度
Dropout 率	0.1	    防止过拟合的正则化参数

###########目录结构描述
├── Readme.md                   
├── src                         // 源代码
│   ├── model.py                // 模型
│   ├── train.py                // 训练脚本
│   ├── test.py                 // 测试脚本
├── data
│   ├── tiny_shakespeare.txt    //数据集
├── results                     //训练结果和参数记录
├── scripts
│   ├── run.sh                  //运行

