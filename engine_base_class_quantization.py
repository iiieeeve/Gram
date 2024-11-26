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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import timm
import timm.optim.optim_factory as optim_factory
assert timm.__version__ == "0.3.2"

from model.modeling_vqkd import VQKD
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import get_grad_norm_
from utils.config import Config, get_param_sets
import utils.lr_sched as lr_sched
from utils.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner
from shock.dataset.dataset import ShockDataset

class TrainBCQ:
    def __init__(self, cf,logger,args,summaryWriter):

        self.logger = logger
        self.summaryWriter = summaryWriter
        self.cf = cf
        self.args = args
        

        self.train()

    @torch.no_grad()
    def calculate_codebook_usage(self, data_loader_list, ch_names_list, test_data_loader_name_list):

        metric_logger = misc.MetricLogger(self.logger,delimiter="  ")
        header = 'Calculating codebook usage:'

        # switch to evaluation mode
        self.model.eval()
        
        codebook_num = self.cf.vocab_size
        codebook_cnt = torch.zeros(codebook_num, dtype=torch.float64).to(self.cf.device)

        for data_loader, ch_names, data_loader_name in zip(data_loader_list, ch_names_list, test_data_loader_name_list):
            for step, (data,_) in enumerate(metric_logger.log_every(data_loader, 10, header)):
                data = data/100
                data = data.type(torch.FloatTensor).to(self.cf.device) #B,C,2000
                data = rearrange(data, 'b c (n t) -> b (n c) t ', t=self.cf.sample_rate) #B nc 200 
                data = data.float().to(self.cf.device, non_blocking=True) / 100

                outputs = self.model_without_ddp.get_tokens(data,ch_names)['token'].view(-1)
                
                outputs_gather_list = [torch.zeros_like(outputs) for _ in range(misc.get_world_size())]
                torch.distributed.all_gather(outputs_gather_list, outputs)
                all_tokens = torch.cat(outputs_gather_list, dim=0).view(-1) # [B * N * Ngpu, ]
                
                codebook_cnt += torch.bincount(all_tokens, minlength=codebook_num)

        # statistic
        zero_cnt = (codebook_cnt == 0).sum() # 0
        print(f"STAT {data_loader_name}: {zero_cnt} tokens ({(zero_cnt / codebook_num) * 100}%) are never used in this codebook.")
    
    @torch.no_grad()
    def evaluate(self, data_loader_list, ch_names_list):

        metric_logger = misc.MetricLogger(self.logger,delimiter="  ")
        header = 'Validation:'

        # switch to evaluation mode
        self.model.eval()

        if self.cf.if_DDP:
            self.model_without_ddp.quantize.reset_cluster_size(self.cf.device)
        else:
            self.model.quantize.reset_cluster_size(self.cf.device)
        if self.logger is not None:
            self.logger.info("Reset the codebook statistic info in quantizer before each epoch")
        
        for data_loader, ch_names in zip(data_loader_list, ch_names_list):
            for step, (data, _) in enumerate(metric_logger.log_every(data_loader, 10, header)): 
                data = data/100
                data = data.type(torch.FloatTensor).to(self.cf.device) #B,C,2000
                data = rearrange(data, 'b c (n t) -> b (n c) t ', t=self.cf.sample_rate) #B nc 200  
                data = data.float().to(self.cf.device, non_blocking=True) / 100
                data_rec, loss, loss_log, indices  = self.model(data, ch_names)

                metric_logger.update(loss=loss.item())

                new_log_loss = {k.split('/')[-1]:v for k, v in loss_log.items() if k not in ['total_loss']}
            metric_logger.update(**new_log_loss)

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        

        try:
            if self.cf.if_DDP:
                codebook_cluster_size = self.model_without_ddp.quantize._codebook.cluster_size
            else:
                codebook_cluster_size = self.model.quantize._codebook.cluster_size
        except:
            if self.cf.if_DDP:
                codebook_cluster_size = self.model_without_ddp.quantize.cluster_size
            else:
                codebook_cluster_size = self.model.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
        test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        test_stat['unused_code']=zero_cnt
        print("Averaged stats:", test_stat)
        if self.logger != None:
            self.logger.info(f'test stat: {test_stat}')

        return test_stat


    with torch.no_grad():
        def get_decode_EEG(self, data_loader_list,ch_names_list):
            metric_logger = misc.MetricLogger(self.logger,delimiter="  ")
            header = 'Validation:'

            # switch to evaluation mode
            self.model.eval()

            if self.cf.if_DDP:
                self.model_without_ddp.quantize.reset_cluster_size(self.cf.device)
            else:
                self.model.quantize.reset_cluster_size(self.cf.device)
            if self.logger is not None:
                self.logger.info("Reset the codebook statistic info in quantizer before each epoch")
            
            for data_loader, ch_names in zip(data_loader_list, ch_names_list):
                for step, (data, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
                    data = data/100
                    data = data.type(torch.FloatTensor).to(self.cf.device) #B,C,2000
                    data = rearrange(data, 'b c (n t) -> b (n c) t ', t=self.cf.sample_rate) #B nc 200  同一时间的channl排列在一起
                    data = data.float().to(self.cf.device, non_blocking=True) / 100
                    data_rec, loss, loss_log, indices  = self.model(data, ch_names)

                    metric_logger.update(loss=loss.item())

                    new_log_loss = {k.split('/')[-1]:v for k, v in loss_log.items() if k not in ['total_loss']}
                metric_logger.update(**new_log_loss)

            # gather the stats from all processes
            metric_logger.synchronize_between_processes()
            

            try:
                if self.cf.if_DDP:
                    codebook_cluster_size = self.model_without_ddp.quantize._codebook.cluster_size
                else:
                    codebook_cluster_size = self.model.quantize._codebook.cluster_size
            except:
                if self.cf.if_DDP:
                    codebook_cluster_size = self.model_without_ddp.quantize.cluster_size
                else:
                    codebook_cluster_size = self.model.quantize.cluster_size
            zero_cnt = (codebook_cluster_size == 0).sum().item()
            test_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            test_stat['unused_code']=zero_cnt
            print("Averaged stats:", test_stat)
            if self.logger != None:
                self.logger.info(f'test stat: {test_stat}')

            return test_stat


    def load_loader(self):
        data_loader_list,data_loader_name_list, data_loader_chName_list = [],[], []   
        for idd, dd in enumerate(self.cf.dataset):
            _dataset = ShockDataset(file_path=Path(self.cf.h5_data_path)/(dd+'.hdf5'),window_size=self.cf.window_size[idd], stride_size=self.cf.stride_size)
            if 'RUIJIN' in dd:
                ch_name = self.cf.RUIJIN
            elif 'TUAB' in dd:
                ch_name = self.cf.TUAB
            elif 'TUSZ' in dd:
                ch_name = self.cf.TUSZ
            else:
                ch_name = _dataset.get_ch_names()
            if self.cf.if_DDP:
                world_size = misc.get_world_size()
                global_rank = misc.get_rank()
                sampler = torch.utils.data.distributed.DistributedSampler(_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
                loader = torch.utils.data.DataLoader(_dataset, batch_size=self.cf.batch_size, sampler=sampler,num_workers=self.cf.num_workers,drop_last=True)
            else:
                loader = torch.utils.data.DataLoader(_dataset, batch_size=self.cf.batch_size,drop_last=True, shuffle=True)
            data_loader_list.append(loader)
            data_loader_name_list.append(dd)
            data_loader_chName_list.append(ch_name)
        return data_loader_list, data_loader_name_list, data_loader_chName_list
            
    def load_test_loader(self,):       
        data_loader_list,data_loader_name_list, data_loader_chName_list = [],[], []   
        for idd, dd in enumerate(self.cf.test_dataset):
            _dataset = ShockDataset(file_path=Path(self.cf.h5_data_path)/(dd+'.hdf5'),window_size=self.cf.test_window_size[idd], stride_size=self.cf.test_stride_size)
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
                sampler = torch.utils.data.distributed.DistributedSampler(_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
                loader = torch.utils.data.DataLoader(_dataset, batch_size=self.cf.batch_size, sampler=sampler,num_workers=self.cf.num_workers,drop_last=True)
            else:
                loader = torch.utils.data.DataLoader(_dataset, batch_size=self.cf.batch_size,drop_last=True, shuffle=True)
            data_loader_list.append(loader)
            data_loader_name_list.append(dd)
            data_loader_chName_list.append(ch_name)
        return data_loader_list, data_loader_name_list, data_loader_chName_list
    
    
    def train(self):

        if self.cf.load_model_path is not None:
            tmp = torch.load(self.cf.load_model_path, map_location='cpu')
            self.pre_cf = tmp['cf']
            if self.logger !=None:
                self.logger.info(f'get pretrain cf from {self.cf.load_model_path}') 


        if self.cf.if_DDP:
            seed = self.cf.seed + misc.get_rank()
        else:
            seed = self.cf.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        data_loader_list, data_loader_name_list, data_loader_chName_list = self.load_loader()
        test_data_loader_list, test_data_loader_name_list, test_data_loader_chName_list = self.load_test_loader()

        if  self.cf.load_model_path is not None:
            self.model = VQKD(self.pre_cf).to(device=self.cf.device)
        else:
            self.model = VQKD(self.cf).to(device=self.cf.device)


        
        if self.cf.if_DDP:
            eff_batch_size = self.cf.batch_size * misc.get_world_size()

            if self.cf.lr is None:  # only base_lr is specified
                self.cf.lr = self.cf.blr * eff_batch_size / 256
        else:
            self.cf.lr = self.cf.blr
            eff_batch_size = self.cf.batch_size       
        self.loss_scaler = NativeScaler() 
        self.optimizer = create_optimizer(self.cf, self.model)


        if self.cf.load_model_path is not None:
            misc.load_model(self.cf, self.cf.load_model_path, self.model, self.optimizer, self.loss_scaler)
            if self.logger !=None:
                self.logger.info('load finish')        

        if self.cf.if_DDP:
            if self.logger !=None:
                self.logger.info('start DDP model')                  
            self.model = DDP(self.model, device_ids=[self.args.gpu], output_device=self.args.gpu)
            self.model_without_ddp = self.model.module

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.logger !=None:
            self.logger.info('number of VQVAE params (M): %.2f' % (n_parameters / 1.e6))

                
        if self.cf.if_val_mode:
            self.calculate_codebook_usage(test_data_loader_list, test_data_loader_chName_list, test_data_loader_name_list)
            return
      
            
        self.all_len = 0
        for tmp in data_loader_list:
            self.all_len += len(tmp) 
        if self.logger !=None:
            self.logger.info(f'start_epoch: {self.cf.start_epoch}')
            self.logger.info("base lr: %.2e" % (self.cf.lr * 256 / eff_batch_size))
            self.logger.info("actual lr: %.2e" % self.cf.lr)
            self.logger.info("effective batch size: %d" % eff_batch_size)
            self.logger.info(f'len(all_data_loader):  {self.all_len}')  
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
        if self.cf.weight_decay_end is None:
            self.cf.weight_decay_end = self.cf.weight_decay
        self.wd_schedule_values = lr_sched.cosine_scheduler(
            self.cf.weight_decay, self.cf.weight_decay_end, self.cf.epochs, num_training_steps_per_epoch)
        print("Max WD = %.7f, Min WD = %.7f" % (max(self.wd_schedule_values), min(self.wd_schedule_values)))

            
        # start_time = time.time()
        
        for epoch in range(self.cf.start_epoch, self.cf.epochs):
            train_stats = self.train_one_epoch(epoch, data_loader_list, data_loader_name_list, data_loader_chName_list)
            test_stats = self.evaluate(test_data_loader_list, test_data_loader_chName_list)
            
            log_stats = {'epoch': epoch,**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()}}

            if self.cf.model_path and misc.is_main_process():
                with open(os.path.join(self.cf.model_path, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                if self.summaryWriter is not None:
                    self.summaryWriter.flush()

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



    def train_one_epoch(self, epoch, data_loader_list, data_loader_name_list, data_loader_chName_list):
        self.model.train()
        metric_logger = misc.MetricLogger(self.logger,delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 5

        if self.cf.if_DDP:
            self.model_without_ddp.quantize.reset_cluster_size(self.cf.device)
        else:
            self.model.quantize.reset_cluster_size(self.cf.device)
        if self.logger is not None:
            self.logger.info("Reset the codebook statistic info in quantizer before each epoch")

        if self.cf.if_DDP:
            global_rank = misc.get_rank()
        else:
            global_rank = 0
        # iter dataloader

        count = -1

        # epoch_indices = torch.tensor([]).to(device=self.cf.device)
        for tmpid, data_loader in enumerate(data_loader_list):   
            if self.cf.if_DDP:
                data_loader.sampler.set_epoch(epoch)
            data_loader_name = data_loader_name_list[tmpid]
            ch_list = data_loader_chName_list[tmpid]
            for data_iter_step, (data, _) in enumerate(metric_logger.log_every(data_loader, print_freq,global_rank, header)):
                # we use a per iteration (instead of per epoch) lr scheduler
                count += 1
                if count >= self.all_len:
                    continue
                it = epoch*self.all_len + count  # global training iteration
                # print(it)
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
                
                with torch.cuda.amp.autocast(enabled=True):
                    data_rec, loss_vq, loss_log, indices = self.model(data, ch_list)
                    
                loss = loss_vq
                # epoch_indices = torch.cat((epoch_indices, indices.flatten())).to(device=self.cf.device)  
                loss_value = loss.item()

                if not math.isfinite(loss_value):
                    try:
                        raise Exception()
                    except:
                        if self.logger is not None: 
                            self.logger.critical("Loss is {}, stopping training".format(loss_value))
                        sys.exit(1)

                loss /= self.cf.accum_iter

                
                self.optimizer.zero_grad()
                grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=self.cf.grad_clip, parameters=self.model.parameters(),
                        update_grad=(count + 1) % self.cf.accum_iter == 0, retain_graph=True, if_step_update=False)

                self.loss_scaler.step(self.optimizer)
                if self.cf.if_gan:
                    self.loss_scaler.step(self.optimizer_d)
                
                self.loss_scaler.update()


                torch.cuda.synchronize()

                metric_logger.update(loss=loss_value)

                new_loss_log = {k:v for k, v in loss_log.items() if k not in ['total_loss']}
                metric_logger.update(**new_loss_log)
                metric_logger.update(grad_norm=grad_norm)
                
                min_lr = 10.
                max_lr = 0.
                for group in self.optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])

                metric_logger.update(lr=max_lr)
                metric_logger.update(min_lr=min_lr)
                
                weight_decay_value = None
                for group in self.optimizer.param_groups:
                    if group["weight_decay"] > 0:
                        weight_decay_value = group["weight_decay"]
                metric_logger.update(weight_decay=weight_decay_value)

                
                loss_value_reduce = misc.all_reduce_mean(loss_value)

                if self.summaryWriter is not None and (count + 1) % self.cf.accum_iter == 0:
                    """ We use epoch_1000x as the x-axis in tensorboard.
                    This calibrates different curves when batch size changes.
                    """
                    epoch_1000x = int((count / self.all_len + epoch) * 1000)
                    self.summaryWriter.add_scalar('train/loss', loss_value_reduce, epoch_1000x)
                    self.summaryWriter.add_scalar('train/rec_loss', new_loss_log['rec_loss'], epoch_1000x)
                    self.summaryWriter.add_scalar('train/quant_loss', new_loss_log['quant_loss'], epoch_1000x)
                    self.summaryWriter.add_scalar('opt/lr', max_lr, epoch_1000x)
                    self.summaryWriter.add_scalar('opt/weight_decay', weight_decay_value, epoch_1000x)
                    self.summaryWriter.add_scalar('opt/grad_norm', grad_norm, epoch_1000x)

        # stat the codebook usage information
        try:
            if self.cf.if_DDP:
                codebook_cluster_size = self.model_without_ddp.quantize._codebook.cluster_size
            else:
                codebook_cluster_size = self.model.quantize._codebook.cluster_size
        except:
            if self.cf.if_DDP:
                codebook_cluster_size = self.model_without_ddp.quantize.cluster_size
            else:
                codebook_cluster_size = self.model.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()

        metric_logger.synchronize_between_processes()
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if self.logger is not None:
            self.logger.info(f"epoch {epoch}, Averaged stats: {str(metric_logger)},Unused code in codebook: {zero_cnt}")

        train_stat['unused_code']=zero_cnt
        return train_stat


