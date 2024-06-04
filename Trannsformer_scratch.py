import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#encoder-decoder architecture
#position encoding is a matrix that encodes the position of the words in the input sequence
''' max_lenght is the maximum length of the input sequence it is used to determine the maximum length of the positional encoding matrix and 
ensure that the positional encoding matrix is not too large'''
'''d_model is the dimension of the model, it is the number of expected features in the input'''

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        #computes the scaling factor
        scaling_factor = -math.log(10000.0) / d_model
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * scaling_factor)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
        
def sclaed_dot_product_attention(query, key, value, mask=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    attention = torch.nn.functional.softmax(scores, dim=-1)
    return torch.matmul(attention, value), attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        def transform(x):
            #reshape the input tensor x to separate the dimensions for the multiple attention heads
            #-1 automatically determines the size of the dimension
            #transpose from [batch_size, seq_len, num_heads, d_k] to [batch_size, num_heads, seq_len, d_k]
            return x.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        query, key, vlaue = [transform(l(x)) for l, x in zip([self.query, self.key, self.value], (query, key, value))]
        x, attention_weights = sclaed_dot_product_attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.out(x)


#feed forward neural network
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)
        
    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))
    
#encoder layer
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = FeedForward(d_model)
        self.layer_norm_1 = nn.LayerNorm(d_model)
        self.layer_norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask):
        attn_output = self.multi_head_attention(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.layer_norm_1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        return self.norm_2(x)
#combine encoder layers to form the encoder

class Encoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len, dropout=0.1):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(d_model)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x

class Transformer (nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len, output_vocab_size, dropout=0.1):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len, dropout)
        self.fc_out = nn.Linear(d_model, output_vocab_size)
    def forward(self, src, src_mask=None):
        enc_output = self.encoder(src, src_mask)
        return self.fc_out(enc_output)
    
num_layers = 6
d_ff = 2028
input_vocab_size = 1000
output_vocab_size = 1000
max_len = 100
dropout = 0.1
num_heads = 8
num_layers = 6
d_model = 512

model = Transformer(num_layers, d_model, num_heads, d_ff, input_vocab_size, max_len, output_vocab_size, dropout)
src = torch.randint(0, input_vocab_size, (32, max_len))
src_mask = None
output = model(src, src_mask)
print(output.shape)