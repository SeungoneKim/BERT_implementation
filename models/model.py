import torch
import torch.nn as nn
import torch.nn.functional as F
from models.embedding import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from models.attention import ScaledDotProductAttention, MultiHeadAttention, FeedForward
from models.layers import EncoderLayer

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, 
                    model_dim, key_dim, value_dim, hidden_dim, 
                    num_head, num_layer, drop_prob, device):
        super(Encoder,self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, model_dim, max_len, drop_prob, device)
        
        self.layers = nn.ModuleList([EncoderLayer(model_dim, key_dim, value_dim, 
                                                  hidden_dim, num_head, 
                                                  drop_prob) for _ in range(num_layer)])
        
    def forward(self, input_ids, attention_mask, token_type_ids):
        input_emb = self.embedding(input_ids, token_type_ids)
        encoder_output = input_emb
        
        for layer in self.layers:
            encoder_output = layer(encoder_output, attention_mask)
        
        return encoder_output

class NaturalLanguageUnderstandingHead(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(NaturalLanguageUnderstandingHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, encoder_output):
        # mask_position = [bs, tgt_size(15% of sent)]
        mlm_prediction = self.softmax(self.linear_layer(encoder_output)) # [bs,sl,vocab_size]
        
        return mlm_prediction

class NextSentencePredictionHead(nn.Module):
    def __init__(self, model_dim):
        super(NextSentencePredictionHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim,2)
        self.softmax = nn.LogSoftmax(dim=-1)
    
    def forward(self, encoder_output):
        cls_representation = encoder_output[:,0,:]      # [bs, model_dim]
        nsp_prediction = self.softmax(self.linear_layer(cls_representation))  # [bs, 2]
        return nsp_prediction

class BERTModel(nn.Module):
    def __init__(self, pad_idx, mask_idx, cls_idx, sep_idx, unk_idx,
                vocab_size, model_dim, key_dim, value_dim, hidden_dim, 
                num_head, num_layer, max_len, drop_prob, device):
        super(BERTModel, self).__init__()
        self.pad_idx = pad_idx
        self.mask_idx = mask_idx
        self.cls_idx = cls_idx
        self.sep_idx = sep_idx
        self.unk_idx = unk_idx
        self.device = device

        self.Encoder = Encoder(vocab_size, max_len, model_dim, key_dim, value_dim, hidden_dim, num_head, num_layer, drop_prob, device)
        self.NLUHead = NaturalLanguageUnderstandingHead(vocab_size, model_dim)
        self.NSPHead = NextSentencePredictionHead(model_dim)

    def forward(self, input_ids, attention_mask, token_type_ids):
        encoder_output = self.Encoder(input_ids, attention_mask, token_type_ids)
        # mlm_output = [bs, sl, vocab_size]
        # nsp_output = [bs, 2]
        mlm_output = self.NLUHead(encoder_output)
        nsp_output = self.NSPHead(encoder_output)

        return (mlm_output,nsp_output)
    
"""
tmp_model = build_model(0, 103, 101, 102, 100, 
                        30000, 512, 64, 64, 2048, 
                        8, 12, 1024, 0.1, 'cuda:0')

params = list(tmp_model.parameters())
print("The number of parameters:",sum([p.numel() for p in tmp_model.parameters() if p.requires_grad]), "elements")

The number of parameters: 60305970 elements
"""
def build_model(pad_idx, mask_idx, cls_idx, sep_idx, unk_idx,
                vocab_size, model_dim, key_dim, value_dim, hidden_dim, 
                num_head, num_layer, max_len, drop_prob, device):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BERTModel(pad_idx, mask_idx, cls_idx, sep_idx, unk_idx,
                vocab_size, model_dim, key_dim, value_dim, hidden_dim, 
                num_head, num_layer, max_len, drop_prob, device)
    

    return model.cuda() if torch.cuda.is_available() else model