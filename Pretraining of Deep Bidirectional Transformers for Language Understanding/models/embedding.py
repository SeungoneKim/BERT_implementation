import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, model_dim):
        super(TokenEmbedding, self).__init__(vocab_size, model_dim, padding_idx=1)

class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len, device):
        super(PositionalEncoding, self).__init__()
        
        self.encoding = torch.zeros(max_len, model_dim, device=device)
        self.encoding.requires_grad = False
        
        pos = torch.arange(0,max_len,device=device).float().unsqueeze(dim=1)
        _2i = torch.arange(0,model_dim,step=2,device=device).float()
        
        # self.encoding = (sequence_length, hidden_size)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i/model_dim)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i/model_dim)))
        
    def forward(self, tensor):
        batch_size, sequence_length = tensor.size()
        
        # (sequence_length, hidden_size)
        return self.encoding[:sequence_length, :]

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, model_dim, max_len, drop_prob, device):
        super(TransformerEmbedding,self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, model_dim)
        self.pos_emb = PositionalEncoding(model_dim, max_len, device)
        self.seg_emb = nn.Embedding(2,model_dim)
        self.tok_drop_out = nn.Dropout(drop_prob)
        self.seg_drop_out = nn.Dropout(drop_prob)
    
    def forward(self, tensor, token_type_ids):
        tok_emb = self.tok_emb(tensor)
        pos_emb = self.pos_emb(tensor)
        seg_emb = self.seg_emb(token_type_ids)
        
        return self.tok_drop_out(tok_emb) + self.seg_drop_out(seg_emb) + pos_emb