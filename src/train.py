import os
import sys
import argparse
import logging
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from configs import get_config, set_random_fixed
from dataloader import get_pretrain_dataloader, get_finetune_dataloader
from utils import load_metric, load_optimizer, load_scheduler, load_lossfn
from model import build_model

class Trainer():
    def __init__(self):
        
        # set parser
        self.args = get_config()

        # set randomness fixed
        self.set_random(516)

        # set logging
        logging.basicConfig(level=logging.INFO)

        # set if pretrain mode or finetune mode & load dataloader
        self.pretrain_mode= False
        self.dataloader = False
        
        if self.args.pretrain_or_finetune is 'pretrain':
            self.pretrain_mode = True
            self.dataloader = get_pretrain_dataloader(self.args.dataset_name, 
                                self.args.category_name, self.args.batch_size, 
                                self.args.percentage)
        else:
            self.dataloader = get_finetune_dataloader(self.args.dataset_name, 
                                self.args.x_name, self.args.y_name, 
                                self.args.batch_size, self.args.percentage)

        # load metric
        self.metric = get_metric(self.args.metric)

        # set parameters needed for training
        self.best_epoch = 0
        self.best_score = 0
        self.least_loss = float('inf')

        self.training_history = []
        self.validation_history = []

        self.batch_size = self.args.batch_size
        self.batch_num = len(self.dataloader)
        self.n_epoch = self.args.epoch
        
        self.lr = self.args.init_lr
        self.eps = self.args.adam_eps
        self.decay = self.args.weight_decay
        self.decay_epoch = self.args.decay_epoch
        self.beta1 = self.args.beta1
        self.beta2 = self.args.beta2

        self.max_len = self.args.max_len


        # build directory to save weights
        self.weightpath = args.weight_path
        os.mkdir(self.weightpath)
        
        # build model
        self.model = build_model(self.args.pad_idx, self.args.pad_idx, self.args.bos_idx, 
                        self.args.vocab_size, self.args.vocab_size, 
                        self.args.model_dim, self.args.key_dim, self.args.value_dim, self.args.hidden_dim, 
                        self.args.num_head, self.args.num_layers, self.args.max_len, self.args.drop_prob)

        # build optimizer
        self.optimizer = load_optimizer(self.model, self.lr, self.decay)
        
        # build scheduler
        self.scheduler = load_scheduler(self.optimizer, self.decay_epoch)
        
        # build lossfn
        self.lossfn = load_lossfn(self.args.lossfn)

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
        

    def pretrain(self):

        # set initial variables for training (outside epoch)
        train_batch_num = len(train_dataloader)
        validation_batch_num = len(val_dataloader)

        best_epoch=0
        best_score=0
        least_loss = float('inf')
        

        for epoch_idx in range(self.n_epoch):
            logging.info('#################################################')
            logging.info(f"Epoch : {epoch_idx} / {n_epoch}")
            logging.info('#################################################')

            #################################################################
            ####################   PRETRAINING PHASE  #######################
            #################################################################
            
            # switch model to train mode
            self.model.train()

            # set initial variables for training (inside epoch)
            training_loss_per_epoch=0.0
            training_score_per_epoch=0.0

            ########################
            #### Training Phase ####
            ########################
            for batch_idx, model_input in tqdm(enumerate(self.dataloader)):
                # move batch of data to gpu
                model_input = model_input.to(device)
                model_input_input_ids, model_input_attention_mask = model_input

                # compute model output and loss
                model_output = self.model(model_input_input_ids, model_input_attention_mask)
                loss = self.lossfn(model_output)

                # clear gradients, and compute gradient with current batch
                optimizer.zero_grad()
                loss.backward()

                # update gradients
                optimizer.step()

                # add loss to training_loss
                training_loss_per_iteration = loss.item()
                training_loss_per_epoch += training_loss_per_iteration

                # Evaluate summaries with period of display_steps
                if (batch_idx+1) % display_step==0 and batch_idx>0:
                    logging.info(f"Training Phase |  Epoch: {epoch_idx+1} |  Step: {batch_idx+1} / {train_batch_num} | loss : {training_loss_per_iteration}")

            # update scheduler
            self.scheduler.step()

            # save training loss of each epoch, in other words, the average of every batch in the current epoch
            training_mean_loss_per_epoch = training_loss_per_epoch / train_batch_num
            training_history.append(training_mean_loss_per_epoch)

    def finetune(self):
        NotImplementedError

    def set_random(seed_num):
        set_random_fixed(seed_num)

