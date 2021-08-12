import os
import sys
import argparse
import logging
from tqdm.notebook import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import transformers
from config.configs import set_random_fixed, get_path_info
from data.dataloader import get_Pretrain_dataloader
from data.tokenizer import Tokenizer
from util.utils import (load_metricfn, load_optimizer, load_scheduler, load_lossfn, 
                        save_checkpoint, load_checkpoint, save_bestmodel, 
                        time_measurement, count_parameters, initialize_weights)
from models.model import build_model

class Pretrain_Trainer():
    def __init__(self, parser):
        
        # set parser
        self.args = parser.parse_args()

        # save loss history to plot later on
        self.training_history = []
        self.validation_history = []

        # set variables needed for training
        self.n_epoch = self.args.epoch
        self.train_batch_size = self.args.train_batch_size
        self.display_step = self.args.display_step # training
        self.val_batch_size = self.args.val_batch_size
        self.test_batch_size = self.args.test_batch_size
        self.display_examples = self.args.display_examples # testing
        
        self.lr = self.args.init_lr
        self.eps = self.args.adam_eps
        self.weight_decay = self.args.weight_decay
        self.beta1 = self.args.adam_beta1
        self.beta2 = self.args.adam_beta2

        self.warmup_steps = self.args.warm_up
        self.factor = self.args.factor
        self.patience = self.args.patience
        self.clip = self.args.clip

        self.language = self.args.language
        self.max_len = self.args.max_len
        self.vocab_size = self.args.vocab_size

        self.next_sent_prob = self.args.pretrain_next_sent_prob
        self.masking_prob = self.args.pretrain_masking_prob
        self.training_ratio = self.args.pretrain_training_ratio
        self.validation_ratio = self.args.pretrain_validation_ratio
        self.test_ratio = self.args.pretrain_test_ratio

        self.device = self.args.device

        # build dataloader
        self.train_dataloader, self.val_dataloader, self.test_dataloader = get_Pretrain_dataloader(
            self.train_batch_size, self.val_batch_size, self.test_batch_size,
            self.language, self.max_len,
            self.args.pretrain_dataset_name, self.args.pretrain_dataset_type, self.args.pretrain_category_name,
            self.next_sent_prob, self.masking_prob, 
            self.training_ratio, self.validation_ratio, self.test_ratio,
            self.args.pretrain_percentage
        )
        self.train_batch_num = len(self.train_dataloader)
        self.val_batch_num = len(self.val_dataloader)
        self.test_batch_num = len(self.test_dataloader)
        
        self.t_total = self.train_batch_num * self.n_epoch
        
        # build model
        self.model= build_model(self.args.pad_idx, self.args.mask_idx, 
                        self.args.cls_idx, self.args.sep_idx, self.args.unk_idx,
                        self.vocab_size, self.args.model_dim, self.args.key_dim, self.args.value_dim, self.args.hidden_dim, 
                        self.args.num_heads, self.args.num_layers, self.max_len, self.args.drop_prob, self.device)
        
        self.model.apply(initialize_weights)

        # build optimizer
        self.optimizer = load_optimizer(self.model, self.lr, self.weight_decay, 
                                        self.beta1, self.beta2, self.eps)
        
        # build scheduler
        self.scheduler = load_scheduler(self.optimizer, self.factor, self.patience)
        
        # build lossfn
        self.mlm_lossfn = load_lossfn(self.args.pretrain_lossfn,self.args.pad_idx)
        self.nsp_lossfn = load_lossfn(self.args.pretrain_lossfn)

    def train_test(self):
        best_model_epoch, training_history, validation_history = self.pretrain()
        self.save_best_pretrained_model(best_model_epoch)
        self.plot(training_history, validation_history)
        
    def pretrain(self):
        
        # set logging        
        logging.basicConfig(level=logging.WARNING)
        
        # logging message
        sys.stdout.write('#################################################\n')
        sys.stdout.write('You have started training the model.\n')
        print('Your model size is : ')
        count_parameters(self.model)
        sys.stdout.write('#################################################\n')

        # set randomness of training procedure fixed
        self.set_random(516)
        
        # build directory to save to model's weights
        self.build_directory()

        # set initial variables for training, validation
        train_batch_num = len(self.train_dataloader)
        validation_batch_num = len(self.val_dataloader)

        # set initial variables for model selection
        best_model_epoch=0
        best_model_mlm_loss = float('inf')
        best_model_nsp_loss = float('inf')
        best_model_total_loss = float('inf')

        # save information of the procedure of training
        training_history=[]
        validation_history=[]

        # predict when training will end based on average time
        total_time_spent_secs = 0
        
        # start of looping through training data
        for epoch_idx in range(self.n_epoch):
            # measure time when epoch start
            start_time = time.time()
            
            sys.stdout.write('#################################################\n')
            sys.stdout.write(f"Epoch : {epoch_idx+1} / {self.n_epoch}")
            sys.stdout.write('\n')
            sys.stdout.write('#################################################\n')

            ########################
            #### Training Phase ####
            ########################
            
            # switch model to train mode
            self.model.train()

            # set initial variables for training (inside epoch)
            training_loss_per_epoch=0.0

            # train model using batch gradient descent with Adam Optimizer
            for batch_idx, batch in tqdm(enumerate(self.train_dataloader)):
                # move batch of data to gpu
                input_ids = batch['input_ids'].to(self.device)              #[bs, sl]
                label_ids = batch['label_ids'].to(self.device)              #[bs, sl]
                attention_mask = batch['attention_mask'].to(self.device)    #[bs, sl]
                token_type_ids = batch['token_type_ids'].to(self.device)    #[bs, sl]
                is_next = batch['is_next'].to(self.device)                  #[bs, 1]

                # compute model output
                # model_output_mlm = [bs, sl, vocab_size]
                # model_output_nsp = [bs, 2]
                model_output_mlm, model_output_nsp = self.model(input_ids, attention_mask, token_type_ids)      # [bs,sl,vocab], [bs,2]

                # reshape model output and labels
                reshaped_model_output_mlm = model_output_mlm.contiguous().view(-1,model_output_mlm.shape[-1])   # [bs*sl,vocab]
                reshaped_label_ids = label_ids.contiguous().view(-1)                                            # [bs*sl]
                reshaped_is_next = is_next.contiguous().view(-1)
                
                # clear graidents
                self.optimizer.zero_grad()
                
                # compute loss using model output for MLM Task & NSP Task
                nsp_loss = self.nsp_lossfn(model_output_nsp, reshaped_is_next)                                               # [bs,2] + [bs,1] => []
                mlm_loss = self.mlm_lossfn(reshaped_model_output_mlm, reshaped_label_ids)
                total_loss = nsp_loss + mlm_loss

                #compute gradient with current batch
                total_loss.backward()

                # clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clip)

                # update gradients
                self.optimizer.step()

                # add loss to training_loss
                training_loss_per_iteration_mlm = mlm_loss.item()
                training_loss_per_iteration_nsp = nsp_loss.item()
                training_loss_per_iteration = total_loss.item()
                training_loss_per_epoch += training_loss_per_iteration

                # Display summaries of training procedure with period of display_step
                if ((batch_idx+1) % self.display_step==0) and (batch_idx>0):
                    sys.stdout.write(f"Training Phase |  Epoch: {epoch_idx+1} |  Step: {batch_idx+1} / {train_batch_num} | MLM loss : {training_loss_per_iteration_mlm} | NSP loss : {training_loss_per_iteration_nsp} | Total loss : {training_loss_per_iteration}")
                    sys.stdout.write('\n')

            # update scheduler
            self.scheduler.step()

            # save training loss of each epoch, in other words, the average of every batch in the current epoch
            training_mean_loss_per_epoch = training_loss_per_epoch / train_batch_num
            training_history.append(training_mean_loss_per_epoch)

            ##########################
            #### Validation Phase ####
            ##########################

            # switch model to eval mode
            self.model.eval()

            # set initial variables for validation (inside epoch)
            validation_loss_per_epoch=0.0 

            # validate model using batch gradient descent with Adam Optimizer
            for batch_idx, batch in tqdm(enumerate(self.val_dataloader)):
                # move batch of data to gpu
                input_ids = batch['input_ids'].to(self.device)              #[bs, sl]
                label_ids = batch['label_ids'].to(self.device)              #[bs, sl]
                attention_mask = batch['attention_mask'].to(self.device)    #[bs, sl]
                token_type_ids = batch['token_type_ids'].to(self.device)    #[bs, sl]
                is_next = batch['is_next'].to(self.device)                  #[bs, 1]

                # compute model output
                # model_output_mlm = [bs, sl, vocab_size]
                # model_output_nsp = [bs, 2]
                model_output_mlm, model_output_nsp = self.model(input_ids, attention_mask, token_type_ids)          # [bs,sl,vocab], [bs,2]
                
                # reshape model output and labels for MLM Task
                reshaped_model_output_mlm = model_output_mlm.contiguous().view(-1,model_output_mlm.shape[-1])       # [bs*sl,vocab]
                reshaped_label_ids = label_ids.contiguous().view(-1)                                                # [bs*sl]
                reshaped_is_next = is_next.contiguous().view(-1)                                                    # [bs]

                # compute loss using model output and labels for NSP Task
                nsp_loss = self.nsp_lossfn(model_output_nsp, reshaped_is_next)
                mlm_loss = self.mlm_lossfn(reshaped_model_output_mlm, reshaped_label_ids)
                total_loss = nsp_loss + mlm_loss
                
                # add loss to training_loss
                validation_loss_per_iteration_mlm = mlm_loss.item()
                validation_loss_per_iteration_nsp = nsp_loss.item()
                validation_loss_per_iteration = total_loss.item()
                validation_loss_per_epoch += validation_loss_per_iteration

            
            # save validation loss of each epoch, in other words, the average of every batch in the current epoch
            validation_mean_loss_per_epoch = validation_loss_per_epoch / validation_batch_num
            validation_history.append(validation_mean_loss_per_epoch)

            # Display summaries of validation result after all validation is done
            sys.stdout.write(f"Validation Phase |  Epoch: {epoch_idx+1} | MLM loss : {validation_loss_per_iteration_mlm} | NSP loss : {validation_loss_per_iteration_nsp} | Total loss : {validation_mean_loss_per_epoch}")
            sys.stdout.write('\n')


            # Model Selection Process using validation_mean_score_per_epoch
            if (validation_mean_loss_per_epoch < best_model_loss):
                best_model_epoch = epoch_idx
                best_model_loss = validation_mean_loss_per_epoch

                save_checkpoint(self.model, self.optimizer, epoch_idx,
                            os.path.join(self.args.weight_path,str(epoch_idx+1)+".pth"))

            # measure time when epoch end
            end_time = time.time()

            # measure the amount of time spent in this epoch
            epoch_mins, epoch_secs = time_measurement(start_time, end_time)
            sys.stdout.write(f"Time spent in {epoch_idx+1} is {epoch_mins} minuites and {epoch_secs} seconds\n")
            
            # measure the total amount of time spent until now
            total_time_spent += (end_time - start_time)
            total_time_spent_mins = int(total_time_spent/60)
            total_time_spent_secs = int(total_time_spent - (total_time_spent_mins*60))
            sys.stdout.write(f"Total amount of time spent until {epoch_idx+1} is {total_time_spent_mins} minuites and {total_time_spent_secs} seconds\n")

            # calculate how more time is estimated to be used for training
            avg_time_spent_secs = total_time_spent_secs / (epoch_idx+1)
            left_epochs = self.n_epoch - (epoch_idx+1)
            estimated_left_time = avg_time_spent_secs * left_epochs
            estimated_left_time_mins = int(estimated_left_time/60)
            estimated_left_time_secs = int(estimated_left_time - (estimated_left_time_mins*60))
            sys.stdout.write(f"Estimated amount of time until {self.n_epoch} is {estimated_left_time_mins} minuites and {estimated_left_time_secs} seconds\n")

        # summary of whole procedure    
        sys.stdout.write('#################################################\n')
        sys.stdout.write(f"Training and Validation has ended.\n")
        sys.stdout.write(f"Your best model was the model from epoch {best_model_epoch} with loss : {best_model_loss}\n")
        sys.stdout.write('#################################################\n')

        return best_model_epoch, training_history, validation_history

    def save_best_pretrained_model(self, best_model_epoch):

        # logging message
        sys.stdout.write('#################################################\n')
        sys.stdout.write('Saving your best Model.\n')
        sys.stdout.write('#################################################\n')

        # set randomness of training procedure fixed
        self.set_random(516)

        # set weightpath
        weightpath = os.path.join(os.getcwd(),'weights')

        # loading the best_model from checkpoint
        best_model = build_model(self.args.pad_idx, self.args.pad_idx, self.args.bos_idx, 
                self.args.vocab_size, self.args.vocab_size, 
                self.args.model_dim, self.args.key_dim, self.args.value_dim, self.args.hidden_dim, 
                self.args.num_head, self.args.num_layers, self.args.max_len, self.args.drop_prob)
        
        load_checkpoint(best_model, self.optimizer, 
                    os.path.join(self.args.weight_path,str(best_model_epoch+1)+".pth"))

        # save best model
        save_bestmodel(best_model,self.optimizer,self.args,
                            os.path.join(self.args.final_model_path,"bestmodel.pth"))

    def plot(self, training_history, validation_history):
        step = np.linspace(0,self.n_epoch,self.n_epoch)
        plt.plot(step,np.array(training_history),label='Training')
        plt.plot(step,np.array(validation_history),label='Validation')
        plt.xlabel('number of epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

        cur_path = os.getcwd()
        save_dir = os.path.join(curpath,'plot')
        path = os.path.join(save_dir, 'train_validation_plot.png')
        sys.stdout.write('Image of train, validation history saved as plot png!\n')
        
        plt.savefig(path)

    def build_directory(self):
        # Making directory to store model pth
        curpath = os.getcwd()
        weightpath = os.path.join(curpath,'weights')
        os.mkdir(weightpath)

    def set_random(self, seed_num):
        set_random_fixed(seed_num)

