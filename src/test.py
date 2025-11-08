import torch
import torch.nn as nn
import os
from model import Transformer

def quick_test():
    """快速测试函数 - 使用正确的参数名称"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 模型文件路径
    model_path = "Transformer/results/transformer_model_final.pth"
    text_path = "Transformer/data/tiny_shakespeare.txt"
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        # 尝试其他可能的路径
        alternative_paths = [
            "results/transformer_model_final.pth",
            "transformer_model_final.pth",
            "Transformer/results/best_model.pth"
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                print(f"找到模型文件: {model_path}")
                break
        else:
            print("请确保模型文件存在")
            return
    
    if not os.path.exists(text_path):
        print(f"错误: 文本文件不存在: {text_path}")
        return
    
    # 加载文本构建词汇表（单词级别，与训练一致）
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 使用单词级别的词汇表（与train.py一致）
    words = text.split()
    unique_words = sorted(list(set(words)))
    vocab_size = len(unique_words)
    word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for idx, word in enumerate(unique_words)}
    
    print(f"词汇表大小: {vocab_size}")
    print(f"前10个单词示例: {unique_words[:10]}")
    
    model_args = {
        'src_vocab_size': vocab_size,
        'tgt_vocab_size': vocab_size,
        'd_model': 128,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 512,
        'dropout': 0.0,           
        'pad_idx': 0
    }
    
    print("\n模型参数:")
    for key, value in model_args.items():
        print(f"  {key}: {value}")
        
    try:
        model = Transformer(**model_args)
        print("✓ 模型创建成功!")
    except TypeError as e:
        print(f"✗ 模型创建失败: {e}")
        return
        
    # 加载模型权重
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 处理不同的checkpoint格式
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("✓ 从检查点加载模型权重")
            else:
                model.load_state_dict(checkpoint)
                print("✓ 直接加载模型权重")
        else:
            model.load_state_dict(checkpoint)
            print("✓ 加载模型权重")
            
    except Exception as e:
        print(f"✗ 加载模型权重失败: {e}")
        return
    
    model.to(device)
    model.eval()
    
    def generate_text(prompt, max_length=50, temperature=0.8, top_k=40):
        """生成文本（单词级别，与train.py一致）"""
        model.eval()
        
        # 将提示文本分割成单词
        start_words = prompt.split()
        tokens = [word_to_idx.get(word, 0) for word in start_words]
        generated_words = start_words.copy()
        
        print(f"提示: '{prompt}'")
        print(f"生成: ", end='', flush=True)
        
        with torch.no_grad():
            for step in range(max_length):
                # 准备输入（取最后seq_len个token）
                seq_len = 128  # 与训练时一致
                if len(tokens) > seq_len:
                    input_tokens = tokens[-seq_len:]
                else:
                    input_tokens = tokens
                
                input_tensor = torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)
                
                # 前向传播
                logits = model(src=None, tgt=input_tensor)
                
                # 获取最后一个token的logits
                next_token_logits = logits[0, -1, :] / max(temperature, 0.1)
                
                # Top-k 采样
                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, len(next_token_logits)))
                    mask = torch.ones_like(next_token_logits, dtype=torch.bool)
                    mask[top_k_indices] = False
                    next_token_logits[mask] = -float('Inf')
                
                # 应用softmax获取概率
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # 从分布中采样
                try:
                    next_token = torch.multinomial(probs, num_samples=1).item()
                except:
                    next_token = torch.argmax(probs).item()
                
                # 将token转换回单词
                next_word = idx_to_word.get(next_token, '<UNK>')
                
                # 打印生成的单词
                print(next_word + ' ', end='', flush=True)
                
                tokens.append(next_token)
                generated_words.append(next_word)
                
                # 停止条件
                if next_word in ['.', '!', '?'] and step > 10:
                    break
                if next_token == 0:  # pad token
                    break
                if len(generated_words) >= max_length + len(start_words):
                    break
        
        print()  # 换行
        return ' '.join(generated_words)
    
    # 测试多个提示（与train.py中的测试提示一致）
    test_prompts = [
        "We are accounted poor citizens",
        "Your suffering in this dearth",
        "Your most grave belly was deliberate",
        "With every minute you do change a mind",
    ]
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}:")
        try:
            result = generate_text(prompt, max_length=50, temperature=0.8)
            print(f"完整结果: {result}")
            print("-" * 50)
        except Exception as e:
            print(f"生成失败: {e}")
            continue
    
    # 测试不同温度的效果
    print("\n" + "="*60)
    print("温度对比测试")
    print("="*60)
    
    test_prompt = "The future of"
    temperatures = [0.5, 0.8, 1.0, 1.2]
    
    for temp in temperatures:
        print(f"\n温度 {temp}:")
        try:
            result = generate_text(test_prompt, max_length=30, temperature=temp)
            print(f"结果: {result}")
        except Exception as e:
            print(f"失败: {e}")
    
    print("\n✓ 测试完成!")

if __name__ == "__main__":
    quick_test()