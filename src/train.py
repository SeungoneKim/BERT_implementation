def train(self):
        
    current_mode='default'
    if self.pretrain_mode is True:
        current_mode = 'PRETRAIN'
    else:
        current_mode = 'FINETUNE'

    logging.info('#################################################')
    logging.info('You have started training the model.')
    logging.info('Your current mode is',current_mode)
    logging.info('#################################################')
    if self.pretrain_mode is True:
        self.pretrain()
    else:
        self.finetune()