import torch
import torch.nn as nn
import torch.nn.functional as F

# pretrained BPE Tokenizer
from transformers import BertTokenizer
"""
tokenizer = Tokenizer()

print(tokenizer.get_vocab_size())
print()
print(tokenizer.get_vocab())

50265
"""
class Tokenizer():
    def __init__(self, language, max_len):
        self.tokenizer = None
        if language == 'de':
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-german-cased")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                                'bos_token':'[CLS]','eos_token':'[SEP]',
                                                'cls_token':'[CLS]','sep_token':'[SEP]',
                                                'mask_token':'[MASK]','unk_token':'[UNK]'})
            
        else:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]',
                                                'bos_token':'[CLS]','eos_token':'[SEP]',
                                                'cls_token':'[CLS]','sep_token':'[SEP]',
                                                'mask_token':'[MASK]','unk_token':'[UNK]'})

        self.bos_token = self.tokenizer.bos_token
        self.eos_token = self.tokenizer.eos_token
        self.sep_token = self.tokenizer.sep_token
        self.cls_token = self.tokenizer.cls_token
        self.unk_token = self.tokenizer.unk_token
        self.pad_token = self.tokenizer.pad_token     
        self.mask_token = self.tokenizer.mask_token
        self.vocab_size = self.tokenizer.vocab_size
        self.supported_max_len = self.tokenizer.model_max_length
        self.max_len = max_len

        if self.max_len > self.supported_max_len:
            assert "The length you have requested is too long."
    
    # function for Masked Language Modeling
    # "ENCODE" = "tokenize" + "convert token to id" + "truncation & padding" + "Transform to Tensor"
    def tokenize(self, batch_sentence):
        return self.tokenizer.tokenize(batch_sentence)

    # function for Masked Language Modeling
    # "ENCODE" = "tokenize" + "convert token to id" + "truncation & padding" + "Transform to Tensor"
    def convert_tokens_to_ids(self, token):
        return self.tokenizer.convert_tokens_to_ids(token)

    # function for Masked Language Modeling
    def get_mask_token(self):
        return self.tokenizer.mask_token

    def get_mask_token_idx(self):
        return self.convert_tokens_to_ids(self.get_mask_token())

    def get_pad_token(self):
        return self.tokenizer.pad_token

    def get_pad_token_idx(self):
        return self.convert_tokens_to_ids(self.get_pad_token())
    
    def encode(self, batch_sentences):
        return self.tokenizer(batch_sentences, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_len)

    def encode_multiple(self, batch_sentences1, batch_sentences2):
        return self.tokenizer(batch_sentences1, batch_sentences2, padding="max_length", truncation=True, return_tensors="pt", max_length=self.max_len)
    
    def encode_into_input_ids(self, batch_sentences):
        return self.encode(batch_sentences)['input_ids']
    
    def encode_multiple_into_input_ids(self, batch_sentences1, batch_sentences2):
        return self.encode_multiple(batch_sentences1, batch_sentences2)['input_ids']
    
    def encode_into_token_type_ids(self, batch_sentences):
        return self.encode(batch_sentences)['token_type_ids']
    
    def encode_multiple_into_token_type_ids(self, batch_sentences1, batch_sentences2):
        return self.encode_multiple(batch_sentences1, batch_sentences2)['token_type_ids']
    
    def encode_into_attention_mask(self, batch_sentence):
        return self.encode(batch_sentence)['attention_mask']
    
    def encode_multiple_into_attention_mask(self, batch_sentences1, batch_sentences2):
        return self.encode_multiple(batch_sentences1, batch_sentences2)['attention_mask']
    
    def decode(self, encoded_input_ids):
        decoded_output=[]
        for batch in encoded_input_ids:
            batch_output=[]
            for ids in batch:
                batch_output.append( [self.tokenizer.decode(ids, skip_special_tokens=True)] )
            decoded_output.append(batch_output)

        return decoded_output
    
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_vocab(self):
        return self.tokenizer.get_vocab()
    
    def add_word(self, word):
        self.tokenizer.add_tokens([word])
        self.vocab_size = len(self.tokenizer)
        return self.vocab_size
    
    def add_words(self, list_of_words):
        self.tokenizer.add_tokens(list_of_words)
        self.vocab_size = len(self.tokenizer)
        return self.vocab_size

    def get_special_tokens(self):
        return [self.tokenizer.pad_token, self.tokenizer.eos_token, self.tokenizer.cls_token,
                self.tokenizer.bos_token, self.tokenizer.sep_token, self.tokenizer.mask_token,
                self.tokenizer.unk_token]

    def get_end_token(self):
        return self.tokenizer.sep_token