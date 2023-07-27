# -*- encoding: utf-8 -*-
'''
Filename         :io.py
Description      :
Time             :2023/07/27 22:27:10
Author           :daiyizheng
Email            :387942239@qq.com
Version          :1.0
'''
from typing import Any, Text, Union, List, Optional

from pathlib import Path
import json

from ruamel import yaml as yaml
import pandas as pd


def _is_ascii(text: Text) -> bool:
    return all(ord(character) < 128 for character in text)

def read_file(filename) -> Any:
    with open(filename, encoding="utf-8") as f:
        return f.read()

def read_yaml(content: Text,
              reader_type: Union[Text, List[Text]] = "safe"
              ) -> Any:
    if _is_ascii(content):
        # Required to make sure emojis are correctly parsed
        content = (
            content.encode("utf-8")
                .decode("raw_unicode_escape")
                .encode("utf-16", "surrogatepass")
                .decode("utf-16")
        )

    yaml_parser = yaml.YAML(typ=reader_type)
    yaml_parser.preserve_quotes = True

    return yaml_parser.load(content) or {}


def read_config_yaml(filename: Union[Text, Path]) -> Any:
    content = read_file(filename)
    return read_yaml(content)

def read_smiles_csv(path):
    return pd.read_csv(path,
                       usecols=['SMILES']).astype(str)['SMILES'].tolist()

def write_to_file(filename: Text, text: Any) -> None:
    """Write a text to a file."""

    write_text_file(str(text), filename)
    
def write_text_file(
        content: Text,
        file_path: Union[Text, Path],
        encoding: Text = "utf-8",
        append: bool = False,
) -> None:
    mode = "a" if append else "w"
    with open(file_path, mode, encoding=encoding) as file:
        file.write(content)    

def write_json_to_file(filename: Text, obj: Any, **kwargs: Any) -> None:
    """Write an object as a json string to a file."""

    write_to_file(filename, json_to_string(obj, **kwargs))
    
def write_yaml(filename:Text, obj:Any, **kwargs:Any)->None:
    with open(filename, 'w', encoding='utf-8') as f:
        yaml.dump(obj, f, Dumper=yaml.RoundTripDumper)



def json_to_string(obj: Any,
                   **kwargs: Any) -> Text:
    indent = kwargs.pop("indent", 2)
    ensure_ascii = kwargs.pop("ensure_ascii", False)
    return json.dumps(obj, indent=indent, ensure_ascii=ensure_ascii, **kwargs)

def read_json(filename: Text) -> Any:
    """Read json from a file."""

    with open(filename, encoding="utf-8") as f:
        return json.load(f)