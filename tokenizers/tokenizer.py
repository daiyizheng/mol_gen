# -*- encoding: utf-8 -*-
'''
Filename         :tokenizer.py
Description      :分词工具
Time             :2023/07/26 15:14:04
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from __future__ import annotations, print_function
import abc
from typing import Dict, Text, Tuple, Union, List

from mio import write_json_to_file, read_json
import torch

class Tokenizer(abc.ABC):
    """
    Tokenizer class for converting smiles to tokens.
    """
    def __init__(self, 
                 vocab_list,
                 bos_token: Text = '<BOS>',
                 eos_token: Text = '<EOS>',
                 pad_token: Text = '<PAD>',
                 unk_token: Text = '<UNK>',
                 **kwargs
                 ):
        self.__vocab_list = vocab_list
        self.__bos_token=bos_token
        self.__eos_token=eos_token
        self.__pad_token=pad_token
        self.__unk_token=unk_token

    @property
    def vocab_list(self) -> list:
        return self.__vocab_list

    @property
    def bos_token_ids(self):
        return self.token_to_id[self.__bos_token]
    
    @property
    def bos_token(self):
        return self.__bos_token
       
    @property
    def eos_token_ids(self):
        return self.token_to_id[self.__eos_token]

    @property
    def eos_token(self):
        return self.__eos_token

    @property
    def unk_token_ids(self):
        return self.token_to_id[self.__unk_token]

    @property
    def unk_token(self):
        return self.__unk_token

    @property
    def pad_token_ids(self):
        return self.token_to_id[self.__pad_token]

    @property
    def pad_token(self):
        return self.__pad_token

    def tokenize(self, smiles: str) -> list:
        """
        Tokenize smiles.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def convert_token_to_ids(self, token:Text):
        """
        Covert token to ids.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def convert_ids_to_token(self, ids:int):
        """
        Covert ids to token.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def save_config(self, path: Text) -> None:
        raise NotImplementedError
    
    @classmethod
    def load_config_from_path(cls, *args, **kwargs):
        raise NotImplementedError
    
    @classmethod
    def load_config(cls, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, smiles: str) -> list:
        return self.tokenize(smiles)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class CharTokenizer(Tokenizer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        all_syms = self.vocab_list + [self.bos_token, self.eos_token, self.pad_token, self.unk_token]
        self.token_to_id = {c: i for i, c in enumerate(all_syms)}
        self.id_to_token = {i: c for i, c in enumerate(all_syms)}
    
    @staticmethod
    def tokenize(smiles: Text
                 ) -> List:
        """
        Tokenize smiles.
        
        Args:
            smiles: smiles string
        Returns:
            tokenized list
        """
        return [c for c in smiles]
  
    @staticmethod
    def from_data(data:Union[List[Text], Tuple],
                  bos_token:Text = '<BOS>', 
                  eos_token:Text = '<EOS>',
                  pad_token:Text = '<PAD>',
                  unk_token:Text = '<UNK>',
                  **kwargs):
        """
        load vocab list from smiles
        Args:
            data: smiles list
            bos_token: <BOS> token
            eos_token: <EOS> token
            pad_token: <PAD> token
            unk_token: <UNK> token
            **kwargs: other arguments
        Returns: Tongkenizer object
        """
        vocab_list = set()
        for string in data:
            vocab_list.update(string)
        vocab_list = sorted(list(vocab_list))
        
        CharTokenizer.check_special_tokens(vocab_list, bos_token, eos_token, pad_token, unk_token)
        config = {
            "vocab_list":vocab_list,
            "bos_token":bos_token,
            "eos_token":eos_token,
            "pad_token":pad_token,
            "unk_token":unk_token
        }
        return CharTokenizer.load_config(config=config, **kwargs)

    def save_config(self, filename: Text) -> None:
        """
        Save config to file.
        Args:
            filename: config file path
        """
        tokenizer_config = {
            "name": self.__class__.__name__,
            "bos_token" : self.bos_token,
            "eos_token" : self.eos_token,
            "pad_token" : self.pad_token,
            "unk_token" : self.unk_token,
            "vocab_list" : self.vocab_list
        }
        ## save to json file
        write_json_to_file(filename=filename, obj=tokenizer_config)

    def convert_token_to_ids(self, token:Text) -> int:
        """
        Covert token to ids.
        
        Args:
            token: token
        Returns:
            ids
        """
        return self.token_to_id.get(token, self.unk_token_ids)

    def convert_ids_to_token(self, ids: int):
        """
        Covert ids to token
        
        Args:
            ids: ids
        Returns:
            token
        """
        return self.id_to_token.get(ids, self.unk_token)
    
    def string_to_ids(self, 
                      string:Text, 
                      is_add_bos_eos_token_ids:bool=True):
        token_ids = [self.convert_token_to_ids(s) for s in string]
        if is_add_bos_eos_token_ids:
            token_ids = [self.bos_token_ids] + token_ids + [self.eos_token_ids]
        return token_ids

    def ids_to_string(self, 
                      ids:int, 
                      is_del_bos_eos_token:bool=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if is_del_bos_eos_token:
            ids = ids[1:]
            ids = ids[:-1]
        return "".join([self.convert_ids_to_token(i) for i in ids])

    @staticmethod
    def check_special_tokens(vocab_list:List, 
                             bos_token:Text, 
                             eos_token:Text, 
                             pad_token:Text,
                             unk_token:Text):
        if (bos_token in vocab_list) or (eos_token in vocab_list) or \
            (pad_token in vocab_list) or (unk_token in vocab_list):
            raise ValueError('SpecialTokens in chars')

    @classmethod
    def load_config_from_path(cls, path):
        config = read_json(path)
        return cls.load_config(config=config)
        
    
    @classmethod
    def load_config(cls, config:Dict, **kwargs) -> Tokenizer:
                             
        """
        Load tokenizer from config file.
        {
            "name":xxTokenizer,
            "bos_token" : "<BOS>",
            "eos_token" : '<EOS>',
            "pad_token" : '<PAD>',
            "unk_token" : '<UNK>',
            "vocab_list" : []
        }
        """
        vocab_list = config["vocab_list"]
        bos_token = config["bos_token"] if config["bos_token"] else None
        eos_token = config["eos_token"] if config["eos_token"] else None
        pad_token = config["pad_token"] if config["pad_token"] else None
        unk_token = config["unk_token"] if config["unk_token"] else None
        cls.check_special_tokens(vocab_list, bos_token, eos_token, pad_token, unk_token)

        return cls(vocab_list=vocab_list, 
                   bos_token=bos_token,
                   eos_token=eos_token,
                   pad_token=pad_token,
                   unk_token=unk_token,
                   **kwargs)
    
    def __len__(self):
        return len(self.token_to_id)

  
# if __name__ == '__main__':
#     import pandas as pd
#     df = pd.read_csv("/DYZ/dyz1/custom_package/drugflow/examples/datasets/train.csv")
#     smiles = df["SMILES"].to_numpy()
#     tokenize = CharTokenizer.from_data(smiles)
#     tokenize.save_config("./test.json")
#     t = CharTokenizer.load_config_from_path("./test.json")
#     print(t)