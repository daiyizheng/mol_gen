# -*- encoding: utf-8 -*-
'''
Filename         :main.py
Description      :
Time             :2023/07/27 22:48:42
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
import sys

import pandas as pd

from component_builder import ComponentBuilder
from utils import set_seed
from mio import read_config_yaml

def train(config):
    train_data_path = "datasets/train.csv"
    val_data_path = "datasets/test.csv"
    trainer = ComponentBuilder().create_component(component_config=config)
    trainer.train(train_data_path, val_data_path)

def sample(config):
    trainer = ComponentBuilder().load_component(component_meta=config)
    smiles = trainer.predict()
    smiles = pd.DataFrame(smiles, columns=['SMILES'])
    smiles.to_csv("results/vae.csv", index=False)

if __name__ == '__main__':
    set_seed(2023)
    path = "/DYZ/dyz1/custom_package/mol_gen/datasets/vae_config.yml"
    config = read_config_yaml(path)
    train(config=config)
    sample(config=config)
    
