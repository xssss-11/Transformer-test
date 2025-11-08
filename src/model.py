import math
import torch
import torch.nn as nn


'''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度
        
        # Q、K、V的线性投影层及输出线性层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(self, Q, K, V, mask=None):
        # Q、K、V的初始形状：(batch, seq_len, d_model)
        batch_size = Q.size(0)
        
        # 线性投影
        Q = self.W_q(Q)  # 形状：(batch, seq_len, d_model)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 拆分多头：重塑并转置，使形状变为(batch, heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力得分：Q @ K^T，形状变为(batch, heads, seq_len, seq_len)，并缩放
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 应用掩码（若有）
        if mask is not None:
            # 将掩码中值为0的位置填充为-∞，经过softmax后概率变为0
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # 计算注意力权重并获取注意力输出
        attn_weights = torch.softmax(scores, dim=-1)  # 形状：(batch, heads, seq_len, seq_len)
        attn_output = torch.matmul(attn_weights, V)    # 形状：(batch, heads, seq_len, d_k)
        
        # 合并多头：转置后重塑，恢复为(batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous() \
                      .view(batch_size, -1, self.d_model)
        
        # 最终线性投影
        output = self.W_o(attn_output)
        return output'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model必须能被num_heads整除"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        # 关键修复：初始化注意力权重
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        nn.init.constant_(self.W_q.bias, 0.0)
        nn.init.constant_(self.W_k.bias, 0.0)
        nn.init.constant_(self.W_v.bias, 0.0)
        nn.init.constant_(self.W_o.bias, 0.0)
    
    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len = Q.size(0), Q.size(1)
        
        # 线性投影
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # 拆分多头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 计算注意力得分
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # 关键修复：更稳定的掩码处理
        if mask is not None:
            # 确保掩码形状正确
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 增加head维度
            # 使用较大的负值而不是 -inf，避免softmax产生NaN
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # 关键修复：稳定的softmax
        attn_weights = torch.softmax(scores, dim=-1)
        
        # 检查注意力权重是否有NaN
        if torch.isnan(attn_weights).any():
            print("警告: 注意力权重包含NaN")
            # 紧急修复：用均匀分布替换NaN
            attn_weights = torch.nan_to_num(attn_weights, nan=1.0/seq_len)
        
        attn_output = torch.matmul(attn_weights, V)
        
        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous() \
                      .view(batch_size, -1, self.d_model)
        
        output = self.W_o(attn_output)
        return output
    
'''class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)    # 第一层线性层（维度扩展）
        self.fc2 = nn.Linear(d_ff, d_model)    # 第二层线性层（维度恢复）
        self.relu = nn.ReLU()                  # ReLU激活函数
        self.dropout = nn.Dropout(dropout)     # Dropout层（正则化）
    
    def forward(self, x):
        # x的初始形状：(batch, seq_len, d_model)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super(PositionalEncoding, self).__init__()
        # 预计算位置编码矩阵，形状：(max_seq_len, d_model)
        pe = torch.zeros(max_seq_len, d_model)
        # 生成位置索引，形状：(max_seq_len, 1)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # 计算分母项：10000^(2i/d_model)的对数，再取指数
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model))
        
        # 偶数索引位置使用正弦函数，奇数索引位置使用余弦函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # 扩展维度，形状变为(1, max_seq_len, d_model)，适配批量输入
        pe = pe.unsqueeze(0)
        # 注册为缓冲（非可学习参数，但随模型保存）
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x的形状：(batch, seq_len, d_model)
        seq_len = x.size(1)
        # 将位置编码添加到token嵌入中，仅取与输入序列长度匹配的部分
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return x

class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        # 关键修复：使用GELU而不是ReLU，数值更稳定
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
        # 权重初始化
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.constant_(self.fc2.bias, 0.0)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        return out
    
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)  # 自注意力层
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层
    
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_out = self.self_attn(x, x, x, mask)  # 形状：(batch, seq_len, d_model)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_out = self.feed_forward(x)  # 形状：(batch, seq_len, d_model)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)   # 掩码自注意力层
        self.cross_attn = MultiHeadAttention(d_model, num_heads)  # 交叉注意力层
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)  # 前馈网络
        self.norm1 = nn.LayerNorm(d_model)  # 第一层归一化
        self.norm2 = nn.LayerNorm(d_model)  # 第二层归一化
        self.norm3 = nn.LayerNorm(d_model)  # 第三层归一化
        self.dropout = nn.Dropout(dropout)  # Dropout层
    
    def forward(self, x, enc_out=None, src_mask=None, tgt_mask=None):
        # 1. 掩码自注意力
        attn_out = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # 2. 交叉注意力（仅当提供编码器输出时执行）
        if enc_out is not None:
            attn_out2 = self.cross_attn(x, enc_out, enc_out, src_mask)
            x = self.norm2(x + self.dropout(attn_out2))
        
        # 3. 前馈网络
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x
    
'''class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4,
                 num_layers=2, d_ff=256, max_seq_len=5000, dropout=0.1, pad_idx=0):
        super(Transformer, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # 源序列与目标序列的嵌入层（指定填充索引）
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        # 位置编码层
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        # Dropout层（嵌入后应用）
        self.dropout = nn.Dropout(dropout)
        
        # 编码器层栈与解码器层栈（通过ModuleList存储多个层）
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终输出投影层（从解码器输出维度d_model投影到目标词汇表大小）
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
    
    def generate_mask(self, src, tgt):
        # 生成源序列掩码（若提供源序列）
        if src is not None:
            # 形状：(batch, 1, 1, src_len)，非填充位置为True
            src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        else:
            src_mask = None
        
        # 生成目标序列掩码（若提供目标序列）
        tgt_mask = None
        if tgt is not None:
            # 目标序列填充掩码：形状(batch, 1, tgt_len, 1)
            tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(3)
            seq_len = tgt.size(1)
            # 未来掩码：上三角矩阵（对角线以上为1，即无效位置），形状(1, seq_len, seq_len)
            nopeak_mask = torch.ones((1, seq_len, seq_len), device=tgt.device).triu(diagonal=1)
            # 反转未来掩码：对角线及以下为True（允许关注的位置），形状(1, 1, seq_len, seq_len)
            nopeak_mask = (nopeak_mask == 0).unsqueeze(1)
            # 结合填充掩码和未来掩码：逻辑与操作
            tgt_mask = tgt_pad_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        # src形状：(batch, src_len)；tgt形状：(batch, tgt_len)；语言建模场景下src可设为None
        # 生成掩码
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 1. 编码器前向传播（若提供源序列）
        if src is not None:
            # 嵌入 + 位置编码 + Dropout
            enc_input = self.dropout(self.positional_encoding(self.src_embedding(src)))
            # 经过所有编码器层
            enc_output = enc_input
            for layer in self.encoder_layers:
                enc_output = layer(enc_output, mask=src_mask)
        else:
            enc_output = None
        
        # 2. 解码器前向传播
        # 嵌入 + 位置编码 + Dropout
        dec_input = self.dropout(self.positional_encoding(self.tgt_embedding(tgt)))
        # 经过所有解码器层
        dec_output = dec_input
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_out=enc_output, 
                               src_mask=src_mask, tgt_mask=tgt_mask)
        
        # 3. 输出投影：生成每个token的logits
        output_logits = self.fc_out(dec_output)  # 形状：(batch, tgt_len, tgt_vocab_size)
        
        return output_logits'''

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=128, num_heads=4,
                 num_layers=2, d_ff=256, max_seq_len=5000, dropout=0.1, pad_idx=0):
        super(Transformer, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # 使用修复后的组件
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)
        
        # 使用修复后的层
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        
        # 关键修复：添加额外的层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        print("初始化稳定模型权重...")
        
        # 嵌入层初始化
        nn.init.normal_(self.src_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.tgt_embedding.weight, mean=0, std=0.02)
        if self.pad_idx is not None:
            nn.init.constant_(self.src_embedding.weight[self.pad_idx], 0)
            nn.init.constant_(self.tgt_embedding.weight[self.pad_idx], 0)
        
        # 输出层初始化
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0.0)
    
    def generate_mask(self, src, tgt):
        # 保持原有实现
        if src is not None:
            src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        else:
            src_mask = None
        
        tgt_mask = None
        if tgt is not None:
            tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(3)
            seq_len = tgt.size(1)
            nopeak_mask = torch.ones((1, seq_len, seq_len), device=tgt.device).triu(diagonal=1)
            nopeak_mask = (nopeak_mask == 0).unsqueeze(1)
            tgt_mask = tgt_pad_mask & nopeak_mask
        
        return src_mask, tgt_mask
    
    def forward(self, src, tgt):
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        
        # 关键修复：嵌入后立即缩放
        if src is not None:
            enc_input = self.src_embedding(src) * math.sqrt(self.d_model)
            enc_input = self.dropout(self.positional_encoding(enc_input))
            enc_output = enc_input
            for layer in self.encoder_layers:
                enc_output = layer(enc_output, mask=src_mask)
        else:
            enc_output = None
        
        # 关键修复：目标嵌入也进行缩放
        dec_input = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        dec_input = self.dropout(self.positional_encoding(dec_input))
        
        dec_output = dec_input
        for layer in self.decoder_layers:
            dec_output = layer(dec_output, enc_out=enc_output, 
                               src_mask=src_mask, tgt_mask=tgt_mask)
        
        # 关键修复：最终层归一化
        dec_output = self.final_norm(dec_output)
        output_logits = self.fc_out(dec_output)
        
        return output_logits