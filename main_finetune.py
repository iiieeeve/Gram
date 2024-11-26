import sys
import os
import numpy as np
from tqdm import tqdm
import argparse
from copy import deepcopy
from pathlib import Path
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from engine_finetune import Finetune
import utils.misc as misc
from utils.config import Config, get_param_sets



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VQGAN")
    parser.add_argument('--local_rank', default=-1, type=int)  
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--port', default=29529, type=int)
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')


    parser.add_argument('--if_DDP',  action='store_true')

    parser.add_argument('--if_finetune',  action='store_true')
    parser.add_argument('--if_scratch',  action='store_true')
    parser.add_argument('--if_change_th',  action='store_true')
    parser.add_argument('--run_escription','-R', default='test', type=str,
                        help='Experiment Description')    
    parser.add_argument('--seed', default=0, type=int, )   
    parser.add_argument('--config_path','-CP', default='./config/TUEV_finetune.yaml', type=str,
                        help='Experiment Description')    
    parser.add_argument('--load_model_path', default='./result/checkpoints/base.pth',type=str,
                        help='Experiment Description')   



    args = parser.parse_args()
    

    config = Config(args.config_path)
    cf_tuning=config.tuning
    cf_fix = config.fix
    cfs = get_param_sets(cf_tuning)
    for k,v in vars(args).items():
        cf_fix[k]=v
    if cf_fix.if_DDP:
        #init DDP
        print('init DDP')
        misc.init_distributed_mode(args)
        print('init DDP finish')
        cf_fix.device = args.device
    
    home_path = Path(cf_fix.home_path)/cf_fix.run_escription/f'seed_{cf_fix.seed}'
    home_path.mkdir(exist_ok=True, parents=True)
    log_path = home_path/f'{cf_fix.run_escription}.log'
    log_path_result = home_path/f'{cf_fix.run_escription}_result.log'
    model_path = home_path/'model'  #no file name 
    cf_fix.model_path = model_path
    model_path.mkdir(exist_ok=True, parents=True)


    # fix the seed for reproducibility
    if cf_fix.if_DDP:
        world_size = misc.get_world_size()
        global_rank = misc.get_rank()
        print(f'global_rank {global_rank}')
        seed = cf_fix.seed + global_rank

    else:
        seed = cf_fix.seed
    cf_fix.seed = seed
    args.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using  multi-GPU.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False

    best_index = -1
    best_val_stats = None
    beat_test_stats = None
    for cfID,cf in enumerate(cfs):
        cfID = args.cfID
        cf.update(cf_fix)
        if cf.if_DDP:
            if global_rank == 0:
                config.save_to_yaml(cf.to_dict(), home_path/f'cfID_{cfID}.yaml')
        else:
            config.save_to_yaml(cf.to_dict(), home_path/f'cfID_{cfID}.yaml')    
        config.save_to_yaml(cf.to_dict(), home_path/f'cfID_{cfID}.yaml')
        
        tensorboard_path = home_path/'tensorboard_log'/f'cf_{cfID}'
        tensorboard_path.mkdir(exist_ok=True, parents=True)
        if cf.if_DDP:
            if global_rank == 0:
                from loguru import logger
                logger.remove()
                logger.add(sys.stderr)
                logger.add(log_path, filter=lambda record: record["extra"]["name"] == "train_log")
                logger.add(log_path_result, filter=lambda record: record["extra"]["name"] == "result_log")
                logger_train = logger.bind(name="train_log")
                logger_result = logger.bind(name="result_log")
                summaryWriter = SummaryWriter(tensorboard_path)
            else:
                logger_train = None
                logger_result = None
                summaryWriter = None
        else:
            from loguru import logger
            logger.remove()
            logger.add(sys.stderr)
            logger.add(log_path, filter=lambda record: record["extra"]["name"] == "train_log")
            logger.add(log_path_result, filter=lambda record: record["extra"]["name"] == "result_log")
            logger_train = logger.bind(name="train_log")
            logger_result = logger.bind(name="result_log")
            summaryWriter = SummaryWriter(tensorboard_path)
        try:
        
            if logger_train !=None:
                logger_train.info('===========================================')
                logger_train.info('start training')
                logger_train.info('===========================================')
            
            ft = Finetune(cfID,cf,args,logger_train,summaryWriter)
            val_stats, test_stats = ft.train()
            
            if val_stats[cf.main_index] >= best_index:
                best_index = val_stats[cf.main_index]
                best_val_stats = val_stats
                beat_test_stats = test_stats
            if logger_result !=None:
                logger_result.info(f'cfID:{cfID}, cf:{cf}')
                logger_result.info(f'val {val_stats}')
                logger_result.info(f'test {test_stats}')
        except:
            if logger_train != None:
                logger_train.exception('error')
            sys.exit(0)
            



    


