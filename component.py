#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/6/25 20:08
# @Author  : Yizheng Dai
# @Email   : 387942239@qq.com
# @File    : component.py
from __future__ import annotations
from typing import Text, Optional, Any, Dict, List
import copy, argparse

import torch

from visualize import LoggingLogger, WandbLogger


def override_defaults(
        defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """
    Returns:
        updated config
    """
    if defaults:
        config = copy.deepcopy(defaults)
    else:
        config = {}

    if custom:
        for key in custom.keys():
            if isinstance(config, argparse.ArgumentParser):
                setattr(config, key, custom[key])
            else:
                if isinstance(config.get(key), dict):
                    config[key].update(custom[key])
                else:
                    config[key] = custom[key]
    return config


class ComponentMetaclass(type):
    """Metaclass with `name` class property."""

    @property
    def name(cls):
        """The name property is a function of the class - its __name__."""

        return cls.__name__


class Component(metaclass=ComponentMetaclass):
    defaults = {}

    def __init__(self,
                 component_config: Optional[Dict[Text, Any]] = None,
                 **kwargs:Any
                 ) -> None:
        if not component_config:
            component_config = {}
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # makes sure the name of the configuration is part of the config
        # this is important for e.g. persistence
        component_config["name"] = self.name
        self.component_config = override_defaults(
            self.defaults, component_config
        )
        if self.component_config["use_wandb"]:
            project = self.component_config["project"]
            wandb_name = self.component_config["wandb_name"]
            wandb_dir = self.component_config["wandb_dir"]
            wandb_notes = self.component_config["wandb_notes"]
            wandb_tags = self.component_config["wandb_tags"]
            self.log = WandbLogger(project=project,
                                   name=wandb_name,
                                   dir=wandb_dir,
                                   notes = wandb_notes,
                                   tags = wandb_tags)
        else:
            self.log = LoggingLogger()

    @property
    def name(self) -> Text:
        """Access the class's property name from an instance."""

        return type(self).name
    
    @staticmethod
    def check_parameter(parameter:dict):
        for key in parameter:
            if parameter[key] is None:
                raise ValueError("check_parameter")

    @classmethod
    def create(cls,
               component_config: Dict[Text, Any],
               **kwargs:Any
               ) -> "Component":
        return cls(component_config=component_config, **kwargs)

    @classmethod
    def load(cls,
             meta: Dict[Text, Any],
             **kwargs:Any
             ) -> "Component":
        return cls(meta, **kwargs)


    def train(self, *args:Any, **kwargs: Any) -> None:
        raise NotImplementedError
    
    def predict(self, *args:Any, **kwargs:Any)-> None:
        raise NotImplementedError

    def save(self, *args:Any, **kwargs:Any) -> Optional[Dict[Text, Any]]:
        ...