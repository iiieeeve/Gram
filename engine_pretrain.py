import sys
import os
import numpy as np
from tqdm import tqdm
import argparse
from copy import deepcopy
from pathlib import Path
import random
import time
import math
import datetime
import json
from einops import rearrange
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import timm
import timm.optim.optim_factory as optim_factory
assert timm.__version__ == "0.3.2"

from model.modeling_Gram import Gram
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import get_grad_norm_
from utils.config import Config, get_param_setss
import utils.lr_sched as lr_sched
from utils.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from shock.dataset.dataset import ShockDataset
from engine_get_dataset import load_dataset 

class TrainGram:
    def __init__(self, cf,logger,args,summaryWriter):

        self.model = Gram(cf).to(device=cf.device)
        self.logger = logger
        self.summaryWriter = summaryWriter
        self.cf = cf
        self.args = args
        self.loss_scaler = NativeScaler() 
        self.train()
        

    def load_loader(self):
        if self.cf.if_plot_decode_EEG:
            if_shuffle=False
        else:
            if_shuffle=True
        data_loader_list,data_loader_name_list, data_loader_chName_list = [],[], []   
        for idd, dd in enumerate(self.cf.dataset):
            _dataset = ShockDataset(file_path=Path(self.cf.h5_data_path)/(dd+'.hdf5'),window_size=self.cf.window_size[idd], stride_size=self.cf.stride_size)

            if 'RUIJIN' in dd:
                ch_name = self.cf.RUIJIN
            elif 'TUAB' in dd:
                ch_name = self.cf.TUAB
            elif 'TUSZ' in dd:
                ch_name = self.cf.TUSZ
            elif 'TUEV' in dd:
                ch_name = self.cf.TUEV
            else:
                ch_name = _dataset.get_ch_names()
            if self.cf.if_DDP:
                world_size = misc.get_world_size()
                global_rank = misc.get_rank()
                sampler = torch.utils.data.distributed.DistributedSampler(_dataset, num_replicas=world_size, rank=global_rank, shuffle=if_shuffle)
                loader = torch.utils.data.DataLoader(_dataset, batch_size=self.cf.batch_size, sampler=sampler,num_workers=self.cf.num_workers,drop_last=True)
            else:
                loader = torch.utils.data.DataLoader(_dataset, batch_size=self.cf.batch_size,drop_last=True, shuffle=if_shuffle)
            data_loader_list.append(loader)
            data_loader_name_list.append(dd)
            data_loader_chName_list.append(ch_name)
        return data_loader_list, data_loader_name_list, data_loader_chName_list



    def train(self):
        data_loader_list, data_loader_name_list, data_loader_chName_list = self.load_loader()
        if self.cf.if_plot_decode_EEG:
            self.plot_decode_EEG(data_loader_list,data_loader_chName_list,data_loader_name_list)
            return
            
        
        if self.cf.if_DDP:
            self.model = DDP(self.model, device_ids=[self.args.gpu], output_device=self.args.gpu)
            self.model_without_ddp = self.model.module
            
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.logger !=None:
            self.logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

        
        if self.cf.if_DDP:
            eff_batch_size = self.cf.batch_size *self.cf.accum_iter* misc.get_world_size()

            if self.cf.lr is None:  # only base_lr is specified
                self.cf.lr = self.cf.blr * eff_batch_size / 256
        else:
            self.cf.lr = self.cf.blr
            eff_batch_size = self.cf.batch_size           

        if self.cf.if_DDP:
            self.optimizer = create_optimizer(self.cf, self.model_without_ddp)
        else:
            self.optimizer = create_optimizer(self.cf, self.model)
        self.optimizer.zero_grad()

        if self.cf.load_model_path is not None:
            if self.cf.if_DDP:
                misc.load_model(self.cf, self.cf.load_model_path, self.model_without_ddp, self.optimizer, self.loss_scaler)
            else:
                misc.load_model(self.cf, self.cf.load_model_path, self.model, self.optimizer, self.loss_scaler)
            print('load finish')


        self.all_len = 0
        for tmp in data_loader_list:
            self.all_len += len(tmp) 
        if self.logger !=None:
            self.logger.info(f'start_epoch: {self.cf.start_epoch}')
            self.logger.info("base lr: %.2e" % (self.cf.lr * 256 / eff_batch_size))
            self.logger.info("actual lr: %.2e" % self.cf.lr)
            self.logger.info("effective batch size: %d" % eff_batch_size)
            self.logger.info(f'len(all_data_loader):  {self.all_len}')  
            self.logger.info(f'accumulate grad iterations:  {self.cf.accum_iter}')
            for id,tmp in enumerate(data_loader_list): 
                self.logger.info(f'len(data_loader_list[{id}]):  {len(tmp)}') 
            self.logger.info(self.cf)
            

        print("Use step level LR scheduler!")
        num_training_steps_per_epoch = self.all_len
        self.lr_schedule_values = lr_sched.cosine_scheduler(
            self.cf.lr, self.cf.min_lr, self.cf.epochs, num_training_steps_per_epoch,
            warmup_epochs=self.cf.warmup_epochs, warmup_steps=self.cf.warmup_steps,
        )
        print("Max WD = %.7f, Min WD = %.7f" % (max(self.lr_schedule_values), min(self.lr_schedule_values)))
        # if self.logger !=None:
            # np.savetxt('/home/zhaoliming/eeg_gpt/lr.txt',self.lr_schedule_values)
        if self.cf.weight_decay_end is None:
            self.cf.weight_decay_end = self.cf.weight_decay
        self.wd_schedule_values = lr_sched.cosine_scheduler(
            self.cf.weight_decay, self.cf.weight_decay_end, self.cf.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(self.wd_schedule_values), min(self.wd_schedule_values)))

        start_time = time.time()
        
        for epoch in range(self.cf.start_epoch, self.cf.epochs):
            train_stats = self.train_one_epoch(epoch, data_loader_list, data_loader_name_list, data_loader_chName_list)
            if self.cf.if_DDP:
                if self.logger!=None:
                    self.logger.info(f'save last model,{self.cf}')
                misc.save_model(
                        self.cf, model_path=self.cf.model_path,model=self.model, model_without_ddp=self.model_without_ddp, optimizer=self.optimizer,
                        loss_scaler=self.loss_scaler, epoch=epoch,if_last=True)
                if self.cf.model_path and (epoch % 10 == 0 or epoch + 1 == self.cf.epochs):
                    misc.save_model(
                        self.cf, model_path=self.cf.model_path,model=self.model, model_without_ddp=self.model_without_ddp, optimizer=self.optimizer,
                        loss_scaler=self.loss_scaler, epoch=epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},'epoch': epoch,}


            if self.cf.model_path and misc.is_main_process():
                with open(os.path.join(self.cf.model_path, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if self.summaryWriter is not None:
                    self.summaryWriter.flush()


        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if self.logger !=None:
            self.logger.info('Training time {}'.format(total_time_str))

    def train_one_epoch(self, epoch, data_loader_list, data_loader_name_list, data_loader_chName_list):
        self.model.train()
        metric_logger = misc.MetricLogger(self.logger,delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('max_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 5
        if self.cf.if_DDP:
            global_rank = misc.get_rank()
        else:
            global_rank = 0
        # iter dataloader

        count = -1
        tmp = []
        for tmpid, data_loader in enumerate(data_loader_list):   
            if self.cf.if_DDP:
                data_loader.sampler.set_epoch(epoch)
            data_loader_name = data_loader_name_list[tmpid]
            ch_list = data_loader_chName_list[tmpid]
            self.optimizer.zero_grad()

            for data_iter_step, (data, _) in enumerate(metric_logger.log_every(data_loader, print_freq,global_rank, header)):
                # we use a per iteration (instead of per epoch) lr scheduler
                count += 1
                if count >= self.all_len:
                    continue
                it = epoch*self.all_len + count  # global training iteration
                print(f'{self.cf.run_escription},{it}')
                # Update LR & WD for the first acc
                if self.lr_schedule_values is not None or self.wd_schedule_values is not None:
                    for i, param_group in enumerate(self.optimizer.param_groups):
                        if self.lr_schedule_values is not None:
                            param_group["lr"] = self.lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                        if self.wd_schedule_values is not None and param_group["weight_decay"] > 0:
                            param_group["weight_decay"] = self.wd_schedule_values[it]
                    
        
                data = data/100
                data = data.type(torch.FloatTensor).to(self.cf.device) #B,C,2000
                data = rearrange(data, 'b c (n t) -> b (n c) t ', t=self.cf.sample_rate) #B nc 200  同一时间的channl排列在一起
                
                with torch.cuda.amp.autocast():
                    cls_loss, mimic_loss, mlm_acc, pred, mask, proj_weights= self.model(data, ch_list)

                if self.cf.if_mimic:
                    loss = 0.5*cls_loss +0.5* mimic_loss
                else:
                    loss = cls_loss

                loss_value = loss.item()
                cls_loss_value = cls_loss.item()
                mimic_loss_value = mimic_loss.item()
                mlm_acc = mlm_acc.item()

                if not math.isfinite(loss_value):
                    try:
                        raise Exception()
                    except:
                        if self.logger is not None:
                            self.logger.critical("Loss is {}, stopping training".format(loss_value))
                        sys.exit(1)

                loss /= self.cf.accum_iter
                grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=self.cf.grad_clip, parameters=self.model.parameters(),
                        update_grad=(count + 1) % self.cf.accum_iter == 0)
                if (count + 1) % self.cf.accum_iter == 0:
                    self.optimizer.zero_grad()
                loss_scale_value = self.loss_scaler.state_dict()["scale"]

                torch.cuda.synchronize()

                metric_logger.update(loss=loss_value)
                metric_logger.update(loss_scale=loss_scale_value)
                metric_logger.update(cls_loss=cls_loss_value)
                metric_logger.update(mimic_loss=mimic_loss_value)
                metric_logger.update(mlm_acc=mlm_acc)
                metric_logger.update(grad_norm=grad_norm)
                min_lr = 10.
                max_lr = 0.
                for group in self.optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(max_lr=max_lr)
                metric_logger.update(min_lr=min_lr)
                lr = self.optimizer.param_groups[0]["lr"]
                metric_logger.update(lr=lr)

                loss_value_reduce = misc.all_reduce_mean(loss_value)
                cls_loss_value_reduce = misc.all_reduce_mean(cls_loss_value)
                mimic_loss_value_reduce = misc.all_reduce_mean(mimic_loss_value)
                mlm_acc_reduce = misc.all_reduce_mean(mlm_acc)

                if self.summaryWriter is not None and (count + 1) % self.cf.accum_iter == 0:
                    """ We use epoch_1000x as the x-axis in tensorboard.
                    This calibrates different curves when batch size changes.
                    """
                    epoch_1000x = int((count / self.all_len + epoch) * 1000)
                    self.summaryWriter.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
                    self.summaryWriter.add_scalar('train/cls_loss', cls_loss_value_reduce, epoch_1000x)
                    self.summaryWriter.add_scalar('train/mimic_loss', mimic_loss_value_reduce, epoch_1000x)
                    self.summaryWriter.add_scalar('train/mlm_acc', mlm_acc_reduce, epoch_1000x)
                    self.summaryWriter.add_scalar('train/dataloader_id', tmpid, epoch_1000x)
                    self.summaryWriter.add_scalar('lr/max_lr', max_lr, epoch_1000x)
                    self.summaryWriter.add_scalar('lr/loss_scale_value', loss_scale_value, epoch_1000x)
                    self.summaryWriter.add_scalar('lr/min_lr', min_lr, epoch_1000x)
                    self.summaryWriter.add_scalar('lr/lr', lr, epoch_1000x)
                    for layer_id,p in zip(self.cf.out_indices, proj_weights):
                        self.summaryWriter.add_scalar(f'proj_weights/layer_{layer_id}', p, epoch_1000x)


        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        if self.logger is not None:
            self.logger.info(f"epoch {epoch},  Averaged stats: {str(metric_logger)}")
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

