# -*- encoding: utf-8 -*-
'''
Filename         :trainer.py
Description      :
Time             :2023/07/25 09:09:58
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import sys, os
from typing import Text, Optional, Dict, List, Any
import logging
import datetime

from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
import torch
from tqdm import tqdm
import torch.optim as optim

from component import Component
from tokenizers import TOKENIZER_CLASS, Tokenizer
from models.vae.model import VAE
from models.vae.layer import CosineAnnealingLRWithRestart, KLAnnealer
from utils import CircularBuffer
from mio import write_yaml, read_smiles_csv

logger = logging.getLogger("__name__")

class VAEGenerate(Component):
    defaults = {
        "name": "VAEGenerate",
        # tokenizer hyperparameters
        "tokenizer_name": "CharTokenizer",
        "tokenizer_path": None,
         
        # training hyperparameters
        "epochs": 10,
        "batch_size": 512,
        "max_length": 100,
        "config_path":None,
        "model_path":None,
        "save_frequency":20,
        
        # training optim hyperparameters
        "clip_grad": 50,
        "kl_start": 0,
        "kl_w_start": 0.0,
        "kl_w_end": 0.05,
        "lr_start": 3 * 1e-4,
        "lr_n_period": 10,
        "lr_n_restarts": 10,
        "lr_n_mult": 1,
        "lr_end": 3 * 1e-4,
        "n_last": 1000,

        # model hyperparameters
        "encoder_hidden_size": 256,
        "encoder_num_layers": 1,
        "encoder_bidirectional": False,
        "encoder_z_liner_dim": 128,
        "decoder_hidden_size": 512,
        "decoder_num_layers": 3,
        "decoder_bidirectional": True,
        "decoder_z_liner_dim": 512,
        "encodr_dropout_rate": 0.5,
        "decoder_dropout_arte": 0,
        "encoder_rnn_type": "gru",
        "decoder_rnn_type": "gru",
        "freeze_embeddings": False,

        # sample hyperparameters
        "n_samples": 10000, # number of samples
        "temp":1.0,
        "wandb_name": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "wandb_dir": sys.path[0],
        "wandb_notes": "baseline",
        "wandb_tags": ["baseline", "vae"]
    }

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 tokenizer:Tokenizer=None,
                 model:Any=None,
                 **kwargs) -> None:
        super(VAEGenerate, self).__init__(component_config=component_config, 
                                          **kwargs)
        self.tokenizer = tokenizer
        self.model = model
    
    def process(self, data):
        data = read_smiles_csv(data)
        return data
    
    def create_tokenizer(self, data)->Tokenizer:
        ## 初始化tokenizer
        path = self.component_config["tokenizer_path"] if self.component_config["tokenizer_path"] else os.path.join(sys.path[0], self.name+"_tokenizer.json")
        tk = TOKENIZER_CLASS[self.component_config["tokenizer_name"]].from_data(data=data)
        tk.save_config(path)
        return tk
    
    def train(self, train_data, val_data=None):
        ## save tokenizer information 
        logger.info("Tokenizer: Initialization {}".format(self.component_config["tokenizer_name"]))
        self.tokenizer = self.create_tokenizer(train_data)
        
        logger.info("Tokenizer: Initialization End...")
             
        logger.info("initializing model")
        self.model = VAE(vocab_size=len(self.tokenizer),
                         encoder_hidden_size=self.component_config["encoder_hidden_size"],
                         encoder_num_layers=self.component_config["encoder_num_layers"],
                         encoder_bidirectional=self.component_config["encoder_bidirectional"],
                         encoder_z_liner_dim=self.component_config["encoder_z_liner_dim"],
                         decoder_hidden_size=self.component_config["decoder_hidden_size"],
                         decoder_num_layers=self.component_config["decoder_num_layers"],
                         decoder_bidirectional=self.component_config["decoder_bidirectional"],
                         decoder_z_liner_dim=self.component_config["decoder_z_liner_dim"],
                         encodr_dropout_rate=self.component_config["encodr_dropout_rate"],
                         decoder_dropout_arte=self.component_config["decoder_dropout_arte"],
                         encoder_rnn_type=self.component_config["encoder_rnn_type"],
                         decoder_rnn_type=self.component_config["decoder_rnn_type"],
                         freeze_embeddings=self.component_config["freeze_embeddings"],
                         pad_token_ids=self.tokenizer.pad_token_ids
                         )
    
        self.model.to(self.device)
        logger.info("load dataset")
        train_loader = self.get_dataloader(train_data, 
                                           batch_size=self.component_config["batch_size"],
                                           data_type="train")
        val_loader = self.get_dataloader(val_data,
                                         batch_size=self.component_config["batch_size"],
                                         data_type="valid") if val_data is not None else None

        optimizer = self.config_optimizer()
        optimizer = optim.Adam(self.get_optim_params(),
                               lr=self.component_config["lr_start"])

        kl_annealer = KLAnnealer(epochs=self.component_config["epochs"],
                                 kl_start=self.component_config["kl_start"],
                                 kl_w_start=self.component_config["kl_w_start"],
                                 kl_w_end=self.component_config["kl_w_end"])
        lr_annealer = CosineAnnealingLRWithRestart(optimizer=optimizer,
                                                   lr_n_period=self.component_config["lr_n_period"],
                                                   lr_n_mult=self.component_config["lr_n_mult"],
                                                   lr_end=self.component_config["lr_end"])
        self.model.zero_grad()
        for epoch in range(self.component_config["epochs"]):
            tqdm_data = tqdm(train_loader, desc='Training (epoch #{})'.format(epoch))
            kl_weight = kl_annealer(epoch)
            self._train_epoch(epoch, tqdm_data, kl_weight, optimizer)
                      
            if val_loader is not None:
                tqdm_data = tqdm(val_loader, desc='Validation (epoch #{})'.format(epoch))
                self.evaulate(tqdm_data, kl_weight, epoch)
                
                
            # Epoch end
            lr_annealer.step()
            
        ## 保存
            if epoch % self.component_config["save_frequency"] == 0:
                model_path = os.path.join(os.path.dirname(self.component_config["model_path"]), 
                                          'checkpoint_{0:03d}.pt'.format(epoch))
                torch.save(self.model.state_dict(), model_path)
        self.save()
        
    def get_parameters(self):
        
        return self.component_config
    
    def save(self):
        
        torch.save(self.model.state_dict(), self.component_config["model_path"])
        write_yaml(self.component_config["config_path"], obj=self.get_parameters())
        return 
    def evaulate(self, val_loader, kl_weight, epoch):
        postfix = self._train_epoch(epoch, val_loader, kl_weight)
         
    def get_optim_params(self):
        return (p for p in self.model.parameters() if p.requires_grad)
    
    def config_optimizer(self):
        return optim.Adam(self.get_optim_params(),  
                          lr=self.component_config["lr_start"])
    
    def get_collate_fn(self):
        def collate(data):
            data.sort(key=len, reverse=True)
            tensors = []
            for string in data:
                ids = self.tokenizer.string_to_ids(string, is_add_bos_eos_token_ids=True)
                tensor = torch.tensor(ids, dtype=torch.long, device=self.device)
                tensors.append(tensor)
            return tensors
        return collate
    
    def _train_epoch(self, epoch, tqdm_data, kl_weight, optimizer=None):
        if optimizer is None:
            self.model.eval()
        else:
            self.model.train()

        kl_loss_values = CircularBuffer(self.component_config["n_last"])
        recon_loss_values = CircularBuffer(self.component_config["n_last"])
        loss_values = CircularBuffer(self.component_config["n_last"])
        for input_batch in tqdm_data:
            input_batch = tuple(data.to(self.device) for data in input_batch)

            # Forward
            kl_loss, recon_loss = self.model(input_batch)
            loss = kl_weight * kl_loss + recon_loss

            # Backward
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.get_optim_params(),
                                self.component_config["clip_grad"])
                optimizer.step()

            # Log
            kl_loss_values.add(kl_loss.item())
            recon_loss_values.add(recon_loss.item())
            loss_values.add(loss.item())
            lr = (optimizer.param_groups[0]['lr']
                  if optimizer is not None
                  else 0)

            # Update tqdm
            kl_loss_value = kl_loss_values.mean()
            recon_loss_value = recon_loss_values.mean()
            loss_value = loss_values.mean()
            postfix = [f'loss={loss_value:.5f}',
                       f'(kl={kl_loss_value:.5f}',
                       f'recon={recon_loss_value:.5f})',
                       f'klw={kl_weight:.5f} lr={lr:.5f}']
            tqdm_data.set_postfix_str(' '.join(postfix))

        postfix = {
            'epoch': epoch,
            'kl_weight': kl_weight,
            'lr': lr,
            'kl_loss': kl_loss_value,
            'recon_loss': recon_loss_value,
            'loss': loss_value,
            'mode': 'Eval' if optimizer is None else 'Train'}

        return postfix
        
    def get_dataloader(self, 
                       data:List, 
                       batch_size:int,
                       num_workers:int=0,
                       drop_last:bool=False,
                       data_type="train"):
        data = self.process(data)
        
        if data_type == "train":
            dataloader = DataLoader(data,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers,
                          drop_last=drop_last,
                          collate_fn=self.get_collate_fn())
        else:
            dataloader = DataLoader(data,
                          batch_size=batch_size,
                          shuffle=False,
                          num_workers=num_workers,
                          drop_last=drop_last,
                          collate_fn=self.get_collate_fn())
        return dataloader
     
    def predict(self):
        self.model.to(self.device)
        self.model.eval()
        samples = []
        n = self.component_config["n_samples"]
        with tqdm(total=self.component_config["n_samples"], 
                  desc='Generating samples') as T:
            current_samples = self.model.sample(
                batch=min(n, self.component_config["batch_size"]), 
                max_length=self.component_config["max_length"],
                bos_id=self.tokenizer.bos_token_ids,
                pad_id=self.tokenizer.pad_token_ids,
                eos_id=self.tokenizer.eos_token_ids,
                z=None,
                temp=self.component_config["temp"]
            )
            current_samples = [self.tokenizer.ids_to_string(ids,is_del_bos_eos_token=True) 
                               for ids in current_samples]
            samples.extend(current_samples)
            n -= len(current_samples)
            T.update(len(current_samples))
        return samples
        
    @classmethod
    def load(cls, meta:Dict[Text, Any], **kwargs):
        tokenizer =TOKENIZER_CLASS[meta["tokenizer_name"]].load_config_from_path(meta["tokenizer_path"])
        model = VAE(vocab_size=len(tokenizer),
                         encoder_hidden_size=meta["encoder_hidden_size"],
                         encoder_num_layers=meta["encoder_num_layers"],
                         encoder_bidirectional=meta["encoder_bidirectional"],
                         encoder_z_liner_dim=meta["encoder_z_liner_dim"],
                         decoder_hidden_size=meta["decoder_hidden_size"],
                         decoder_num_layers=meta["decoder_num_layers"],
                         decoder_bidirectional=meta["decoder_bidirectional"],
                         decoder_z_liner_dim=meta["decoder_z_liner_dim"],
                         encodr_dropout_rate=meta["encodr_dropout_rate"],
                         decoder_dropout_arte=meta["decoder_dropout_arte"],
                         encoder_rnn_type=meta["encoder_rnn_type"],
                         decoder_rnn_type=meta["decoder_rnn_type"],
                         freeze_embeddings=meta["freeze_embeddings"],
                         pad_token_ids=tokenizer.pad_token_ids
                         )
        model.load_state_dict(torch.load(meta["model_path"]))
        return cls(component_config=meta, tokenizer=tokenizer, model=model, **kwargs)
 