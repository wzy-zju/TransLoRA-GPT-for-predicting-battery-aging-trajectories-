import numpy as np
import torch
import torch.nn as nn
from torch import optim

import threading
import time
from torch.utils.tensorboard import SummaryWriter
from tensorboard.program import TensorBoard
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
# from transformers import BertTokenizer, BertModel
from einops import rearrange
# from embed import DataEmbedding, DataEmbedding_wo_time
import yaml
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import os
import sys
os.chdir(sys.path[0])

from src.data.data_process import load_split_dataloader
from src.utils import *
from src.training_test.train_test import *
from src.gpt2_utils.gpt2_book import GPTModel, GPTModel_wzy
from src.gpt2_utils.gpt_download import download_and_load_gpt2
from src.gpt2_utils.load_weight import load_weights_into_gpt


class GPT2(nn.Module): 
    def __init__(self, configs, device, global_mean, global_std):
        super().__init__()
        #Here you need to download the model yourself and put it in the /root/1_wzy_gpt_weight folder
        settings, params = download_and_load_gpt2(model_size="124M", models_dir="/root/1_wzy_gpt_weight")
        self.gpt2 = GPTModel_wzy(configs)
        load_weights_into_gpt(self.gpt2, params, config)
        gpt_requires_grad(config, self.gpt2)

        range = round((configs['data_process']['soc_range'][1]-configs['data_process']['soc_range'][0])*100)
        self.in_layer = nn.Linear(configs['num_features'] * range, 768)

        if config['data_process']['one_step']:
            self.out_layer_1 = nn.Linear(768*config['data_process']['window_size'], configs['pred_len'])        
            self.out_layer_2 = nn.Linear(configs['num_features'] * range, 1) 


        else :
            self.out_layer_1 = nn.Linear(768, configs['num_features'] * range)
            self.out_layer_2 = nn.Linear(configs['num_features'] * range, 1)



        mean_reshaped = torch.tensor(global_mean, dtype=torch.float32).reshape(1, 1, configs['num_features'], 1)
        std_reshaped = torch.tensor(global_std, dtype=torch.float32).reshape(1, 1, configs['num_features'], 1)
        self.register_buffer('global_mean', mean_reshaped)
        self.register_buffer('global_std', std_reshaped)

    def forward(self, x_uiui, x_uisoh):
        B, W, F, L = x_uiui.shape
        if config['normalied']:
            # x_uiui = (x_uiui - self.global_mean)/(self.global_std + 1e-5)
            raise ValueError("x_uiui is not normalized")

        x_uiui = rearrange(x_uiui, 'b q f s -> b q (f s)')
        outputs = self.in_layer(x_uiui)#outputs (bsz,windows,768)
        outputs = self.gpt2(outputs)

        if config['data_process']['one_step']:
            # outputs = outputs[:, -1, :]
            # outputs = self.out_layer_1(outputs)#outputs (bsz,pred-len)

            outputs = outputs.reshape(B,-1)
            outputs = self.out_layer_1(outputs)#outputs (bsz,pred-len)

            return outputs, None
        else: 
            # During training, autoregressive training relies on output_ui; when predicting soh, it relies on the given x_uisoh
            # When predicting, it is connected

            output_ui = self.out_layer_1(outputs) #(bsz,windows,2*40)
            if config['train']:

                if config['normalied']:
                    #Standardize for UI
                    # x_uisoh = (x_uisoh - self.global_mean)/(self.global_std+1e-5)
                    raise ValueError("x_uisoh is not normalized")



                x_uisoh = rearrange(x_uisoh, 'b q f s -> b q (f s)')

                output_soh = self.out_layer_2(x_uisoh)
            else:
                output_soh = self.out_layer_2(output_ui)

            output_ui = output_ui.reshape(B, W, F, L)
            if config['normalied']:
                output_ui = output_ui * self.global_std
                output_ui = output_ui + self.global_mean

            return output_ui, output_soh.squeeze()
            # 16,50,2,20 ; 16,50



# --- Main program entry ---

if __name__ == '__main__':    
    config_dir = "src/config/config.yaml" 
    config = yaml.safe_load(open(config_dir, "r", encoding='utf-8'))
    set_global_seed(config['seed'])
    

    if config['train']:
        if  config['autotrain']:
            for dataset_info in config['data_process']['sets_to_process']:
                
                if not dataset_info['files']:
                    continue
                dataset_name = dataset_info['name']
                all_files = dataset_info['files']
                print(f"\n{'='*20} Starting to process dataset: {dataset_name} {'='*20}")

            for test_file_id in all_files:
                
                # --- a. Dynamically modify config to split training and test sets for current round ---
                # Training set is all files minus the current test file
                train_file_ids = [f for f in all_files if f != test_file_id]
                # Temporarily modify config dictionary in memory
                config['data_process']['train_sets'] = [{'name': dataset_name, 'files': train_file_ids}]
                config['data_process']['test_sets'] = [{'name': dataset_name, 'files': [test_file_id]}]
                print(f"\n--- [Running Round] Test Set: {dataset_name}_{test_file_id} | Training Set: {[f'{dataset_name}_{f}' for f in train_file_ids]} ---")
                # --- b. Execute your original, complete "training-test" process here ---
                # Reset random seed to ensure reproducibility of each training round
                set_global_seed(config['seed'])
                # **Training part**
                print("\n[Phase 1: Load Data and Training]")

                data_pack = load_split_dataloader(config)

                global_mean = data_pack['global_mean']
                global_std = data_pack['global_std']

                model = GPT2(config, config['device'], global_mean, global_std)
                model = model.to(config['device'])
                optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
                total_params, trainable_params, trainable_percentage = count_parameters(model)
                writer = SummaryWriter("/root/tf-logs", flush_secs=10)

                
                if config['data_process']['one_step']:
                    train_test_one_step(config, model, optimizer, writer, data_pack,total_params, trainable_params, trainable_percentage)
                else: 
                    # train_test_autoregression(config, model, optimizer, writer, data_pack)
                    raise ValueError("autoregression is not implemented")
        else:
            data_pack = load_split_dataloader(config)
            global_mean = data_pack['global_mean']
            global_std = data_pack['global_std']

            model = GPT2(config, config['device'], global_mean, global_std)
            model = model.to(config['device'])
            optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
            total_params, trainable_params, trainable_percentage = count_parameters(model)
            writer = SummaryWriter("/root/tf-logs", flush_secs=10)

            
            if config['data_process']['one_step']:
                train_test_one_step(config, model, optimizer, writer, data_pack,total_params, trainable_params, trainable_percentage)
            else: 
                # train_test_autoregression(config, model, optimizer, writer, data_pack)
                raise ValueError("autoregression is not implemented")
            
    else: #Code before revision
        data_pack = load_split_dataloader(config)

        global_mean = data_pack['global_mean']
        global_std = data_pack['global_std']

        model = GPT2(config, config['device'], global_mean, global_std)
        model = model.to(config['device'])
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'])
        total_params, trainable_params, trainable_percentage = count_parameters(model)
        writer = SummaryWriter("/root/tf-logs", flush_secs=10)

        
        if config['data_process']['one_step']:
            train_test_one_step(config, model, optimizer, writer, data_pack,total_params, trainable_params, trainable_percentage)
        else: 
            # train_test_autoregression(config, model, optimizer, writer, data_pack)
            raise ValueError("autoregression is not implemented")


    print("Training completed")

    

