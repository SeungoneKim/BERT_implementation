import torch
import torch.nn as nn
import torch.nn.functional as F
from embedding import TokenEmbedding, PositionalEncoding, TransformerEmbedding
from attention import ScaledDotProductAttention, MultiHeadAttention, FeedForward
from layers import EncoderLayer, DecoderLayer

class Encoder(nn.Module):
    def __init__(self, vocab_size, max_len, 
                    model_dim, key_dim, value_dim, hidden_dim, 
                    num_head, num_layer, drop_prob, device):
        super(Encoder,self).__init__()
        self.embedding = TransformerEmbedding(vocab_size, model_dim, max_len, drop_prob, device)
        
        self.layers = nn.ModuleList([EncoderLayer(model_dim, key_dim, value_dim, 
                                                  hidden_dim, num_head, 
                                                  drop_prob) for _ in range(num_layer)])
        
    def forward(self, input_ids, token_type_ids, attention_mask):
        input_emb = self.embedding(input_ids)
        encoder_output = input_emb
        
        for layer in self.layers:
            encoder_output = layer(encoder_output, attention_mask)
        
        return encoder_output

class NaturalLanguageUnderstandingHead(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(NaturalLanguageUnderstandingHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim, vocab_size)
    
    def forward(self, encoder_output):
        return F.log_softmax(self.linear_layer(encoder_output),dim=-1) # [bs,sl,vocab_size]

class NextSentencePredictionHead(nn.Module):
    def __init__(self):
        super(NextSentencePredictionHead,self).__init__()
        self.linear_layer = nn.Linear(model_dim,2)
    
    def forward(self, encoder_output):
        output = self.linear_layer(encoder_output) # [bs,sl,2]
        return output[:,0,:] # [bs,2] -> [CLS] token also performs as a sentence embedding

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
        self.NSPHead = NextSentencePredictionHead()

    def forward(self, input_ids, token_type_ids, attention_mask):
        encoder_output = self.Encoder(input_ids, token_type_ids, attention_mask)
        mlm_output = self.NLUHead(encoder_output)
        nsp_output = self.NSPHead(encoder_output)

        return (mlm_output,nsp_output)
    
    # applying mask(opt) : 0s are where we apply masking
    def generate_padding_mask(self, query, key, query_pad_type=None, key_pad_type=None):
        # query = (batch_size, query_length)
        # key = (batch_size, key_length)
        query_length = query.size(1)
        key_length = key.size(1)
        
        # convert query and key into 4-dimensional tensor
        # query = (batch_size, 1, query_length, 1) -> (batch_size, 1, query_length, key_length)
        # key = (batch_size, 1, 1, key_length) -> (batch_size, 1, query_length, key_length)
        query = query.ne(query_pad_idx).unsqueeze(1).unsqueeze(3)
        query = query.repeat(1,1,1,key_length)
        key = key.ne(key_pad_idx).unsqueeze(1).unsqueeze(2)
        key = key.repeat(1,1,query_length,1)
        
        # create padding mask with key and query
        mask = key & query
        
        return mask
    
"""
tmp_model = build_model(0,103,101,102,100,30000,
                       512,64,64,2048,8,12,1024,0.1,'cuda:0')

params = list(tmp_model.parameters())
print("The number of parameters:",sum([p.numel() for p in tmp_model.parameters() if p.requires_grad]), "elements")

The number of parameters: 60304944 elements
"""
def build_model(pad_idx, mask_idx, cls_idx, sep_idx, unk_idx,
                vocab_size, model_dim, key_dim, value_dim, hidden_dim, 
                num_head, num_layer, max_len, drop_prob, device):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BertModel(pad_idx, mask_idx, cls_idx, sep_idx,
                vocab_size, model_dim, key_dim, value_dim, hidden_dim, 
                num_head, num_layer, max_len, drop_prob, device)
    
    return model.cuda() if torch.cuda.is_available() else model