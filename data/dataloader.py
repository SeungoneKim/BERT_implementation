def get_Pretrain_dataloader(train_batch_size, val_batch_size, test_batch_size,
                language, max_len, 
                dataset_name, dataset_type, split_type, category_type, next_sent_prob,
                percentage=None):

    dataset = PretrainDataset_total(language, max_len, 
                dataset_name, dataset_type, split_type, category_type, next_sent_prob
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

def get_Finetune_dataloader(train_batch_size, val_batch_size, test_batch_size,
                language, max_len, 
                dataset_name, dataset_type, split_type, category_type, 
                percentage=None):

    dataset = FinetuneDataset_total(language, max_len, 
                dataset_name, dataset_type, split_type, category_type, 
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