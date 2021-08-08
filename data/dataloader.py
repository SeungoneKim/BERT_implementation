from torch.utils.data import DataLoader, Dataset
from data.dataset import PretrainDataset, PretrainDataset_total

"""
book_train, book_val, book_test = get_Pretrain_dataloader(3,3,3,'en',128,'bookcorpus','plain_text','text',0.5,0.15,0.8,0.1,0.1)
"""
def get_Pretrain_dataloader(train_batch_size, val_batch_size, test_batch_size,
                language, max_len, 
                dataset_name, dataset_type, category_type, next_sent_prob, masking_prob, 
                training_ratio, validation_ratio, test_ratio, percentage=None):

    dataset = PretrainDataset_total(language, max_len, 
                dataset_name, dataset_type, category_type, next_sent_prob, masking_prob,
                training_ratio, validation_ratio, test_ratio, percentage)
    
    train_dataloader = DataLoader(dataset=dataset.getTrainData(),
                            batch_size=train_batch_size,
                            shuffle=True)

    val_dataloader = DataLoader(dataset=dataset.getValData(),
                            batch_size=val_batch_size,
                            shuffle=True)      
    
    test_dataloader = DataLoader(dataset=dataset.getTestData(),
                            batch_size=test_batch_size,
                            shuffle=True)      
    
    return train_dataloader, val_dataloader, test_dataloader

def get_Finetune_dataloader(train_batch_size, val_batch_size, test_batch_size,
                language, max_len, 
                dataset_name, dataset_type, category_type, 
                percentage=None):

    dataset = FinetuneDataset_total(language, max_len, 
                dataset_name, dataset_type, category_type, 
                percentage)
    
    train_dataloader = DataLoader(dataset=dataset.getTrainData(),
                            batch_size=train_batch_size,
                            shuffle=True)

    val_dataloader = DataLoader(dataset=dataset.getValData(),
                            batch_size=val_batch_size,
                            shuffle=True)      
    
    test_dataloader = DataLoader(dataset=dataset.getTestData(),
                            batch_size=test_batch_size,
                            shuffle=True)      
    
    return train_dataloader, val_dataloader, test_dataloader