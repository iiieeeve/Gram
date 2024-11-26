import itertools
import random
import numpy as np
from pathlib import Path
from addict import Dict
from copy import deepcopy
import yaml

import math


def get_param_sets(config:Dict, *, dosage:str='greed', sample_n:int=1) -> list:

    cfs = []
    iter_paras_name = []
    iter_paras_value = []
    cf = Dict()
    for key, value in config.items():
        if isinstance(value, Dict) and 'flag' in value.keys():
            assert value.flag in ['arange', 'linspace']

            if value.flag == 'arange':
                iter_paras_name.append(key)
                iter_paras_value.append(list(np.arange(value.start, value.stop, value.step)))
            if value.flag == 'linspace':
                iter_paras_name.append(key)
                iter_paras_value.append(list(np.linspace(value.start, value.stop, value.number)))

        if isinstance(value, list):
            iter_paras_name.append(key)
            iter_paras_value.append(value)
        else:
            cf[key] = value
    try:
        assert len(iter_paras_name) == len(iter_paras_value)
    except:
        print('Something wrong with the iterable para in config')


    iter_paras = list(itertools.product(*iter_paras_value, repeat=1))
    if dosage == 'random':
        iter_paras = random.sample(iter_paras, sample_n)
    for iter_para in iter_paras:
        for name, value in zip(iter_paras_name, iter_para):
            cf[name] = value
        cfs.append(deepcopy(cf)) 
    return cfs

class Config(Dict):
    '''
    
    '''
    def __init__(self, config_path:Path):

        super().__init__()
        self._load_config(config_path)

    def append(self, config_path:Path):

        self._load_config(config_path)

    def _load_config(self, config: Path):
        with open(config) as f:
            settings = Dict(yaml.load(f, Loader=yaml.FullLoader))
        self.update(settings)

    def save_to_yaml(self, data, save_path):

        with open(save_path, "w") as f:
            yaml.dump(data, f)
