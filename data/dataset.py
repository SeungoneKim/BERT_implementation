from datasets import load_dataset, list_datasets # huggingface library
from tokenizer import Tokenizer

class PretrainDataset(Dataset):
    def __init__(self, language, max_len, 
                dataset_name, dataset_type, split_type, category_type, 
                percentage=None):
        
        if dataset_name not in list_datasets():
            assert('Not available in HuggingFace datasets')
        
        if dataset_type is None:
            if percentage is None:
                data = load_dataset(dataset_name, split=split_type)
            else:
                data = load_dataset(dataset_name, split=f'{split_type}[:{percentage}%]')
        else:
            if percentage is None:
                data = load_dataset(dataset_name, dataset_type, split=split_type)
            else:
                data = load_dataset(dataset_name, dataset_type, split=f'{split_type}[:{percentage}%]')
        
        self.data = data[category_type]
        self.data_len = len(data)
        self.tokenizer = Tokenizer(language, max_len)
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        encoded_data = self.tokenizer.encode(self.data[index])
        return (encoded_data['input_ids'], encoded_data['token_type_ids'], 
                                            encoded_data['attention_mask'])

class PretrainDataset_total():
    def __init__(self, language, max_len, 
                dataset_name, dataset_type, split_type, category_type, 
                percentage=None):
        self.traindata = PretrainDataset_train(dataset_name, category_name, percentage)
        self.valdata = PretrainDataset_val(dataset_name, category_name, percentage)
        self.testdata = PretrainDataset_test(dataset_name, category_name, percentage)
    
    def getTrainData(self):
        return self.traindata
    
    def getValData(self):
        return self.valdata
    
    def getTestData(self):
        return self.testdata

class FineTuneDataset(Dataset):
    def __init__(self, language, max_len, 
                dataset_name, dataset_type, split_type, category_type, 
                x_name, y_name, percentage=None):
        
        if dataset_name not in list_datasets():
            assert('Not available in HuggingFace datasets')
        
        if percentage is None:
            data = load_dataset(dataset_name, dataset_type, split=split_type)
        else:
            data = load_dataset(dataset_name, dataset_type, split=f'{split_type}[:{percentage}%]')

        self.x_name = x_name
        self.y_name = y_name
        self.data_len = len(data) # number of data

        self.data = data[category_type]
        self.dataX = train_data[x_name]
        self.dataY = train_data[y_name]
        
        self.tokenizer = Tokenizer(language, max_len)
        
    def __len__(self):
        return self.data_len
    
    def __getitem__(self, index):
        encoded_datax = self.tokenizer.encode(self.dataX[index])
        encoded_datay = self.tokneizer.encode(self.dataY[index])

        batch ={}
        batch['encoder_input_ids'] = encoded_datax.input_ids
        batch['encoder_attention_mask'] = encoded_datax.attention_mask # will be generated in model as well
        batch['decoder_input_ids'] = encoded_datay.input_ids
        batch['labels'] = encoded_datay.input_ids.clone()
        batch['decoder_attention_mask'] = encoded_datay.attention_mask # will be generated in model as well
        
        return batch


class FineTuneDataset_total():
    def __init__(self, language, max_len, 
                dataset_name, dataset_type, split_type, category_type, 
                x_name, y_name, percentage=None):
        self.traindata = FineTuneDataset(language, max_len, 
                        dataset_name, dataset_type, 'train', category_type, 
                        x_name, y_name, percentage)
        self.valdata = FineTuneDataset(language, max_len, 
                        dataset_name, dataset_type, 'validation', category_type, 
                        x_name, y_name, percentage)
        self.testdata = FineTuneDataset(language, max_len, 
                        dataset_name, dataset_type, 'test', category_type, 
                        x_name, y_name, percentage)
    
    def getTrainData(self):
        return self.traindata
    
    def getValData(self):
        return self.valdata
    
    def getTestData(self):
        return self.testdata