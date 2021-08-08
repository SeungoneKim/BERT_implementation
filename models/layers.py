import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import ScaledDotProductAttention, MultiHeadAttention, FeedForward

class EncoderLayer(nn.Module):
    def __init__(self, model_dim, key_dim, value_dim, hidden_dim, num_head, drop_prob):
        super(EncoderLayer,self).__init__()
        
        self.attention = MultiHeadAttention(model_dim, key_dim, value_dim, num_head)
        self.normalization1 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(drop_prob)
        
        self.ffn = FeedForward(model_dim, hidden_dim, drop_prob)
        self.normalization2 = nn.LayerNorm(model_dim)
        self.dropout2 = nn.Dropout(drop_prob)
        
    def forward(self, tensor, source_mask):
        residual = tensor
        tensor = self.attention(query=tensor,key=tensor,value=tensor,mask=source_mask)
        tensor = self.dropout1(self.normalization1(tensor+residual))
        
        residual = tensor
        tensor = self.ffn(tensor)
        tensor = self.dropout2(self.normalization2(tensor+residual))
        
        return tensor