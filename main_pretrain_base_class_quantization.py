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

from engine_base_class_quantization import TrainBCQ
import utils.misc as misc
from utils.config import Config, get_param_sets



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="BCQ")
    parser.add_argument('--local_rank', default=-1, type=int)   
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--port', default=29529, type=int)
    
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')

    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--if_DDP',  action='store_true')
    parser.add_argument('--if_val_mode',  action='store_true')
    parser.add_argument('--run_escription','-R', default='test', type=str,
                        help='Experiment Description')    
    parser.add_argument('--config_path','-CP', default='./config/vqgan_gpt_base.yaml', type=str,
                        help='Experiment Description')    
    parser.add_argument('--load_model_path', type=str,
                        help='Experiment Description')   
    parser.add_argument('--regress_target', default='raw', type=str,
                        help='raw  or stft') 
    parser.add_argument('--rec_loss_type', default='cosine', type=str,
                        help='mse  or cosine') 
    parser.add_argument('--if_gan' ,  action='store_true') 


    args = parser.parse_args()
    
    #init config
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
    
    home_path = Path(cf_fix.home_path)/cf_fix.run_escription
    home_path.mkdir(exist_ok=True, parents=True)
    log_path = home_path/f'{cf_fix.run_escription}.log'
    model_path = home_path/'model'  #no file name 
    cf_fix.model_path = model_path
    model_path.mkdir(exist_ok=True, parents=True)
    tensorboard_path = home_path/'tensorboard_log'
    tensorboard_path.mkdir(exist_ok=True, parents=True)

    
    if cf_fix.if_DDP:
        #init logger and tensorboard
        world_size = misc.get_world_size()
        global_rank = misc.get_rank()
        if global_rank == 0:
            from loguru import logger
            logger.remove()
            logger.add(sys.stderr)
            logger.add(log_path)
            summaryWriter = SummaryWriter(tensorboard_path)
        else:
            logger = None
            summaryWriter = None
    else:
        from loguru import logger
        logger.remove()
        logger.add(sys.stderr)
        logger.add(log_path)
        summaryWriter = SummaryWriter(tensorboard_path)



    for cf in cfs:
        cf.update(cf_fix)
        if cf.if_DDP:
            if global_rank == 0:
                config.save_to_yaml(cf.to_dict(), home_path/'cf.yaml')
        else:
            config.save_to_yaml(cf.to_dict(), home_path/'cf.yaml')    
        config.save_to_yaml(cf.to_dict(), home_path/'cf.yaml')
        try:
            if logger !=None:
                logger.info(cf.dataset)

            if logger !=None:
                logger.info('===========================================')
                logger.info('start training')
                logger.info('===========================================')
            
            train_transformer = TrainBCQ(cf,logger,args,summaryWriter)
        
        except:
            if logger != None:
                logger.exception('error')
            sys.exit(0)



    


