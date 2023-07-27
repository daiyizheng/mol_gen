# -*- encoding: utf-8 -*-
'''
Filename         :__init__.py
Description      :
Time             :2023/07/27 22:37:51
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from tokenizers.tokenizer import CharTokenizer, Tokenizer


TOKENIZER_CLASS = {
    "CharTokenizer": CharTokenizer
}

__all__ = [TOKENIZER_CLASS, Tokenizer]