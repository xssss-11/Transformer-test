# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, TensorDataset
# from model import Transformer
# import torch.optim as optim
# import requests
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import random

# # ==================== 随机种子设置 ====================
# def set_seed(seed=42):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     # 确保确定性计算
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     print(f"随机种子设置为: {seed}")

# # 设置随机种子
# set_seed(42)

# # --------------------------
# # 1. 超参数设置
# # --------------------------
# seq_len = 128
# batch_size = 32
# n_epochs = 50
# lr = 1e-5
# dropout = 0.1
# d_model = 128
# num_heads = 8
# num_layers = 4
# d_ff = 512
# pad_idx = 0

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"使用设备: {device}")

# file_path = "Transformer/data/tiny_shakespeare.txt"  

# if os.path.exists(file_path):
#     with open(file_path, 'r', encoding='utf-8') as f:
#         text = f.read()
#     print(f"从本地文件读取成功，长度: {len(text)} 字符")
# else:
#     print(f"错误：文件 {file_path} 不存在")
#     exit(1)

# # 分割数据集
# split_idx = int(0.9 * len(text))
# train_text = text[:split_idx]
# val_text = text[split_idx:]

# print(f"训练集长度: {len(train_text)} 字符")
# print(f"验证集长度: {len(val_text)} 字符")

# # 创建单词映射
# words = text.split()
# unique_words = sorted(list(set(words)))
# vocab_size = len(unique_words)
# print(f"词汇表大小: {vocab_size}")

# # 创建单词到索引的映射和索引到单词的映射
# word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
# idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
# # 使用正确的变量名
# stoi = word_to_idx  
# itos = idx_to_word  

# def create_sequences(text, seq_len, stoi):
#     """创建训练序列"""
#     words = text.split()
#     data = torch.tensor([word_to_idx.get(word, 0) for word in words], dtype=torch.long)

#     # 创建序列
#     num_sequences = len(data) // seq_len
#     total_length = num_sequences * seq_len
#     data = data[:total_length]
    
#     # 重塑为序列
#     sequences = data.view(num_sequences, seq_len)
    
#     # 输入和目标 (teacher forcing)
#     inputs = sequences[:, :-1]
#     targets = sequences[:, 1:]
    
#     return inputs, targets

# print("=== 处理训练数据 ===")
# train_inputs, train_targets = create_sequences(train_text, seq_len, stoi)
# val_inputs, val_targets = create_sequences(val_text, seq_len, stoi)

# print(f"训练集：{train_inputs.shape} | 验证集：{val_inputs.shape}")

# # 创建DataLoader（包含随机种子设置）
# def seed_worker(worker_id):
#     worker_seed = torch.initial_seed() % 2**32
#     np.random.seed(worker_seed)
#     random.seed(worker_seed)

# g = torch.Generator()
# g.manual_seed(42)

# train_data = TensorDataset(train_inputs, train_targets)
# val_data = TensorDataset(val_inputs, val_targets)
# train_loader = DataLoader(
#     train_data, 
#     batch_size=batch_size, 
#     shuffle=True, 
#     drop_last=True,
#     worker_init_fn=seed_worker,
#     generator=g
# )
# val_loader = DataLoader(
#     val_data, 
#     batch_size=batch_size, 
#     shuffle=False, 
#     drop_last=False
# )

# # --------------------------
# # 模型初始化与稳定性改进
# # --------------------------
# model = Transformer(
#     src_vocab_size=vocab_size,
#     tgt_vocab_size=vocab_size,
#     d_model=d_model,
#     num_heads=num_heads,
#     num_layers=num_layers,
#     d_ff=d_ff,
#     dropout=dropout,
#     pad_idx=pad_idx
# ).to(device)

# # 自定义权重初始化 - 防止梯度爆炸
# def init_weights(m):
#     if isinstance(m, nn.Linear):
#         nn.init.xavier_uniform_(m.weight, gain=0.1)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0)
#     elif isinstance(m, nn.Embedding):
#         nn.init.normal_(m.weight, mean=0, std=0.01)
# model.apply(init_weights)

# # 损失函数和优化器
# criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
# optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.98), eps=1e-9)

# # 学习率调度器
# scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# # 参数计数
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"可训练参数总数: {total_params}")

# # 记录损失
# train_losses = []
# val_losses = []

# # --------------------------
# # 训练循环
# # --------------------------
# best_val_loss = float('inf')
# patience = 8
# patience_counter = 0

# # 创建结果目录
# os.makedirs("Transformer/results", exist_ok=True)

# print("🚀 开始训练...")
# for epoch in range(1, n_epochs + 1):
#     # 训练阶段
#     model.train()
#     total_train_loss = 0.0
#     train_batches = 0
    
#     for batch_idx, (X, Y) in enumerate(train_loader):
#         X, Y = X.to(device), Y.to(device)
    
#         # 前向传播
#         logits = model(src=None, tgt=X)
#         logits_flat = logits.view(-1, vocab_size)
#         Y_flat = Y.view(-1)
    
#         loss = criterion(logits_flat, Y_flat)
    
#         # 检查NaN
#         if torch.isnan(loss) or torch.isinf(loss):
#             print(f"跳过有问题的批次 {batch_idx}, 损失: {loss.item()}")
#             continue
            
#         # 反向传播
#         optimizer.zero_grad()
#         loss.backward()
        
#         # 梯度裁剪 - 防止梯度爆炸
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
#         # 检查梯度
#         total_norm = 0.0
#         for p in model.parameters():
#             if p.grad is not None:
#                 param_norm = p.grad.data.norm(2)
#                 total_norm += param_norm.item() ** 2
#         total_norm = total_norm ** 0.5
        
#         if total_norm > 1000:
#             print(f"梯度爆炸: {total_norm:.2f}, 跳过更新")
#             continue
            
#         optimizer.step()
        
#         total_train_loss += loss.item()
#         train_batches += 1
        
#         if batch_idx % 50 == 0:
#             print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Grad Norm: {total_norm:.2f}")
    
#     if train_batches == 0:
#         print(f"Epoch {epoch}: 没有有效的训练批次")
#         continue
        
#     avg_train_loss = total_train_loss / train_batches
#     train_losses.append(avg_train_loss)
    
#     # 验证阶段
#     model.eval()
#     total_val_loss = 0.0
#     val_batches = 0
    
#     with torch.no_grad():
#         for Xv, Yv in val_loader:
#             Xv, Yv = Xv.to(device), Yv.to(device)
            
#             logits = model(src=None, tgt=Xv)
#             loss = criterion(logits.view(-1, vocab_size), Yv.view(-1))
            
#             if not torch.isnan(loss) and not torch.isinf(loss):
#                 total_val_loss += loss.item()
#                 val_batches += 1
    
#     if val_batches > 0:
#         avg_val_loss = total_val_loss / val_batches
#         val_losses.append(avg_val_loss)
        
#         # 早停机制
#         if avg_val_loss < best_val_loss:
#             best_val_loss = avg_val_loss
#             torch.save(model.state_dict(), "Transformer/results/best_model.pth")
#             patience_counter = 0
#             print(f"新的最佳验证损失: {best_val_loss:.4f}")
#         else:
#             patience_counter += 1
#             if patience_counter >= patience:
#                 print(f"早停: 验证损失在 {patience} 个epoch内没有改善")
#                 break
#     else:
#         avg_val_loss = float('inf')
#         val_losses.append(avg_val_loss)
#         print("验证阶段所有批次都出现问题")
    
#     # 更新学习率
#     scheduler.step(avg_val_loss)
#     current_lr = optimizer.param_groups[0]['lr']
    
#     print(f"Epoch {epoch:2d}: 训练损失 = {avg_train_loss:.4f}, 验证损失 = {avg_val_loss:.4f}, LR = {current_lr:.2e}")
    
#     # 保存检查点
#     if epoch % 5 == 0:
#         checkpoint_path = f"Transformer/results/checkpoint_epoch{epoch}.pth"
#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'train_loss': avg_train_loss,
#             'val_loss': avg_val_loss,
#             'random_seed': 42
#         }, checkpoint_path)
#         print(f"检查点已保存: {checkpoint_path}")

# # 加载最佳模型
# if os.path.exists("Transformer/results/best_model.pth"):
#     model.load_state_dict(torch.load("Transformer/results/best_model.pth"))
#     print("加载最佳模型完成")

# # 最终保存
# final_model_path = "Transformer/results/transformer_model_final.pth"
# torch.save(model.state_dict(), final_model_path)
# print(f"模型已保存至: {final_model_path}")

# # 绘制损失曲线
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(train_losses) + 1), train_losses, 'o-', color='orange', label='Train loss')
# if len(val_losses) == len(train_losses):
#     plt.plot(range(1, len(val_losses) + 1), val_losses, 'o-', color='red', label='Validation loss')
# plt.title('Transformer')
# plt.xlabel('raining rounds')
# plt.ylabel('loss')
# plt.legend()
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.savefig("Transformer/results/loss_curve.png", dpi=300, bbox_inches='tight')
# plt.close()
# print("训练曲线已保存至: Transformer/results/loss_curve.png")

# # 测试模型生成能力
# def generate_text(model, start_text, max_length=50, temperature=0.8, top_k=40):
#     """改进的文本生成函数"""
#     model.eval()
    
#     # 将起始文本分割成单词
#     start_words = start_text.split()
#     tokens = [word_to_idx.get(word, 0) for word in start_words]
    
#     print(f"起始tokens: {tokens}")
#     print(f"起始单词: {start_words}")
    
#     generated_words = start_words.copy()
    
#     with torch.no_grad():
#         for step in range(max_length):
#             # 准备输入（取最后seq_len个token）
#             if len(tokens) > seq_len:
#                 input_tokens = tokens[-seq_len:]
#             else:
#                 input_tokens = tokens
            
#             input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
            
#             # 前向传播
#             logits = model(src=None, tgt=input_tensor)
            
#             # 获取最后一个token的logits
#             next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
            
#             # 修正的Top-k过滤
#             if top_k > 0:
#                 top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
#                 mask = torch.ones_like(next_token_logits, dtype=torch.bool)
#                 mask[top_k_indices] = False
#                 next_token_logits[mask] = -float('Inf')
            
#             # 应用softmax得到概率
#             probs = torch.softmax(next_token_logits, dim=-1)
            
#             # 从分布中采样
#             try:
#                 next_token = torch.multinomial(probs, num_samples=1).item()
#             except:
#                 next_token = torch.argmax(probs).item()
            
#             # 将token转换回单词
#             next_word = idx_to_word.get(next_token, '<UNK>')  
#             tokens.append(next_token)
#             generated_words.append(next_word)
            
#             # 停止条件
#             if next_word in ['.', '!', '?'] and step > 10:
#                 break
#             if next_token == 0:
#                 break
#             if len(generated_words) >= max_length:
#                 break
    
#     generated_text = ' '.join(generated_words)
#     return generated_text
# test_prompts = [
#     "We are accounted poor citizens",
#     "To be or not to be",
#     "Shall I compare thee"
# ]
# for i, prompt in enumerate(test_prompts, 1):
#     print(f"\n测试 {i}: '{prompt}'")
#     try:
#         generated = generate_text(model, prompt, max_length=30, temperature=0.8)
#         print(f"生成结果: {generated}")
#         print("-" * 60)
#     except Exception as e:
#         print(f"生成失败: {e}")
#         continue

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model import Transformer
import torch.optim as optim
import requests
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import argparse

# ==================== 解析命令行参数 ====================
def parse_args():
    parser = argparse.ArgumentParser(description='Transformer 训练脚本')
    
    # 训练参数
    parser.add_argument('--seq_len', type=int, default=128, help='序列长度')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--n_epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    
    # 模型参数
    parser.add_argument('--d_model', type=int, default=128, help='模型维度')
    parser.add_argument('--num_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=4, help='编码器/解码器层数')
    parser.add_argument('--d_ff', type=int, default=512, help='前馈网络维度')
    
    # 其他参数
    parser.add_argument('--data_path', type=str, default="Transformer/data/tiny_shakespeare.txt", help='数据文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--patience', type=int, default=8, help='早停耐心值')
    parser.add_argument('--save_dir', type=str, default="Transformer/results", help='保存目录')
    
    return parser.parse_args()

# ==================== 随机种子设置 ====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # 确保确定性计算
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"随机种子设置为: {seed}")

# 主函数
def main():
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # --------------------------
    # 超参数设置
    # --------------------------
    seq_len = args.seq_len
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    lr = args.lr
    dropout = args.dropout
    d_model = args.d_model
    num_heads = args.num_heads
    num_layers = args.num_layers
    d_ff = args.d_ff
    pad_idx = 0
    patience = args.patience
    save_dir = args.save_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 打印所有参数
    print("=" * 50)
    print("训练参数:")
    for arg in vars(args):
        print(f"  {arg}: {getattr(args, arg)}")
    print("=" * 50)

    file_path = args.data_path

    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        print(f"从本地文件读取成功，长度: {len(text)} 字符")
    else:
        print(f"错误：文件 {file_path} 不存在")
        exit(1)

    # 分割数据集
    split_idx = int(0.9 * len(text))
    train_text = text[:split_idx]
    val_text = text[split_idx:]

    print(f"训练集长度: {len(train_text)} 字符")
    print(f"验证集长度: {len(val_text)} 字符")

    # 创建单词映射
    words = text.split()
    unique_words = sorted(list(set(words)))
    vocab_size = len(unique_words)
    print(f"词汇表大小: {vocab_size}")

    # 创建单词到索引的映射和索引到单词的映射
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
    # 使用正确的变量名
    stoi = word_to_idx  
    itos = idx_to_word  

    def create_sequences(text, seq_len, stoi):
        """创建训练序列"""
        words = text.split()
        data = torch.tensor([word_to_idx.get(word, 0) for word in words], dtype=torch.long)

        # 创建序列
        num_sequences = len(data) // seq_len
        total_length = num_sequences * seq_len
        data = data[:total_length]
        
        # 重塑为序列
        sequences = data.view(num_sequences, seq_len)
        
        # 输入和目标 (teacher forcing)
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        
        return inputs, targets

    print("=== 处理训练数据 ===")
    train_inputs, train_targets = create_sequences(train_text, seq_len, stoi)
    val_inputs, val_targets = create_sequences(val_text, seq_len, stoi)

    print(f"训练集：{train_inputs.shape} | 验证集：{val_inputs.shape}")

    # 创建DataLoader（包含随机种子设置）
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(args.seed)

    train_data = TensorDataset(train_inputs, train_targets)
    val_data = TensorDataset(val_inputs, val_targets)
    train_loader = DataLoader(
        train_data, 
        batch_size=batch_size, 
        shuffle=True, 
        drop_last=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=batch_size, 
        shuffle=False, 
        drop_last=False
    )

    # --------------------------
    # 模型初始化与稳定性改进
    # --------------------------
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        pad_idx=pad_idx
    ).to(device)

    # 自定义权重初始化 - 防止梯度爆炸
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0, std=0.01)
    model.apply(init_weights)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2, betas=(0.9, 0.98), eps=1e-9)

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    # 参数计数
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数总数: {total_params}")

    # 记录损失
    train_losses = []
    val_losses = []

    # --------------------------
    # 训练循环
    # --------------------------
    best_val_loss = float('inf')
    patience_counter = 0

    # 创建结果目录
    os.makedirs(save_dir, exist_ok=True)

    print("🚀 开始训练...")
    for epoch in range(1, n_epochs + 1):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        train_batches = 0
        
        for batch_idx, (X, Y) in enumerate(train_loader):
            X, Y = X.to(device), Y.to(device)
        
            # 前向传播
            logits = model(src=None, tgt=X)
            logits_flat = logits.view(-1, vocab_size)
            Y_flat = Y.view(-1)
        
            loss = criterion(logits_flat, Y_flat)
        
            # 检查NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"跳过有问题的批次 {batch_idx}, 损失: {loss.item()}")
                continue
                
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪 - 防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # 检查梯度
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 1000:
                print(f"梯度爆炸: {total_norm:.2f}, 跳过更新")
                continue
                
            optimizer.step()
            
            total_train_loss += loss.item()
            train_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, Grad Norm: {total_norm:.2f}")
        
        if train_batches == 0:
            print(f"Epoch {epoch}: 没有有效的训练批次")
            continue
            
        avg_train_loss = total_train_loss / train_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for Xv, Yv in val_loader:
                Xv, Yv = Xv.to(device), Yv.to(device)
                
                logits = model(src=None, tgt=Xv)
                loss = criterion(logits.view(-1, vocab_size), Yv.view(-1))
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_val_loss += loss.item()
                    val_batches += 1
        
        if val_batches > 0:
            avg_val_loss = total_val_loss / val_batches
            val_losses.append(avg_val_loss)
            
            # 早停机制
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f"{save_dir}/best_model.pth")
                patience_counter = 0
                print(f"新的最佳验证损失: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停: 验证损失在 {patience} 个epoch内没有改善")
                    break
        else:
            avg_val_loss = float('inf')
            val_losses.append(avg_val_loss)
            print("验证阶段所有批次都出现问题")
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:2d}: 训练损失 = {avg_train_loss:.4f}, 验证损失 = {avg_val_loss:.4f}, LR = {current_lr:.2e}")
        
        # 保存检查点
        if epoch % 5 == 0:
            checkpoint_path = f"{save_dir}/checkpoint_epoch{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'random_seed': args.seed,
                'args': args
            }, checkpoint_path)
            print(f"检查点已保存: {checkpoint_path}")

    # 加载最佳模型
    if os.path.exists(f"{save_dir}/best_model.pth"):
        model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
        print("加载最佳模型完成")

    # 最终保存
    final_model_path = f"{save_dir}/transformer_model_final.pth"
    torch.save(model.state_dict(), final_model_path)
    print(f"模型已保存至: {final_model_path}")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'o-', color='orange', label='Train loss')
    if len(val_losses) == len(train_losses):
        plt.plot(range(1, len(val_losses) + 1), val_losses, 'o-', color='red', label='Validation loss')
    plt.title('Transformer')
    plt.xlabel('Training rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f"{save_dir}/loss_curve.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"训练曲线已保存至: {save_dir}/loss_curve.png")

    # 测试模型生成能力
    def generate_text(model, start_text, max_length=50, temperature=0.8, top_k=40):
        """改进的文本生成函数"""
        model.eval()
        
        # 将起始文本分割成单词
        start_words = start_text.split()
        tokens = [word_to_idx.get(word, 0) for word in start_words]
        
        print(f"起始tokens: {tokens}")
        print(f"起始单词: {start_words}")
        
        generated_words = start_words.copy()
        
        with torch.no_grad():
            for step in range(max_length):
                # 准备输入（取最后seq_len个token）
                if len(tokens) > seq_len:
                    input_tokens = tokens[-seq_len:]
                else:
                    input_tokens = tokens
                
                input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
                
                # 前向传播
                logits = model(src=None, tgt=input_tensor)
                
                # 获取最后一个token的logits
                next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                
                # 修正的Top-k过滤
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
                    mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                    mask[top_k_indices] = False
                    next_token_logits[mask] = -float('Inf')
                
                # 应用softmax得到概率
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # 从分布中采样
                try:
                    next_token = torch.multinomial(probs, num_samples=1).item()
                except:
                    next_token = torch.argmax(probs).item()
                
                # 将token转换回单词
                next_word = idx_to_word.get(next_token, '<UNK>')  
                tokens.append(next_token)
                generated_words.append(next_word)
                
                # 停止条件
                if next_word in ['.', '!', '?'] and step > 10:
                    break
                if next_token == 0:
                    break
                if len(generated_words) >= max_length:
                    break
        
        generated_text = ' '.join(generated_words)
        return generated_text

    test_prompts = [
        "We are accounted poor citizens",
        "To be or not to be",
        "Shall I compare thee"
    ]
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}: '{prompt}'")
        try:
            generated = generate_text(model, prompt, max_length=30, temperature=0.8)
            print(f"生成结果: {generated}")
            print("-" * 60)
        except Exception as e:
            print(f"生成失败: {e}")
            continue

if __name__ == "__main__":
    main()