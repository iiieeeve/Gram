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
import h5py
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import timm
from timm.loss import LabelSmoothingCrossEntropy
# import timm.optim.optim_factory as optim_factory
assert timm.__version__ == "0.3.2"
from pyhealth.metrics import multiclass_metrics_fn, binary_metrics_fn
import sklearn.metrics as sklearn_metrics

from engine_get_dataset import load_dataset
from model.transformer import VQGANTransformer
from model.modeling_Gram_finetune import Gram
import utils.misc as misc
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from utils.misc import get_grad_norm_
from utils.config import Config, get_param_sets
import utils.lr_sched as lr_sched

from utils.optim_factory import create_optimizer, get_parameter_groups, LayerDecayValueAssigner


from timm.utils import ModelEma



class Finetune:
    def __init__(self, cfID,cf,args,logger,summaryWriter):
        self.logger = logger
        self.summaryWriter = summaryWriter
        self.cf = cf
        self.cfID = cfID
        self.args = args
        self.loss_scaler = NativeScaler() 
        self.criterion = LabelSmoothingCrossEntropy(smoothing=cf.smoothing) if cf.smoothing>0. else nn.CrossEntropyLoss()
        self.metrics = cf.mertric
        self.main_index = cf.main_index
        self.optimizer = None

    def _get_optim(self):
        if self.cf.if_DDP:
            skip_weight_decay_list = self.model_without_ddp.no_weight_decay()
            num_layers = self.model_without_ddp.get_num_layers()
        else:
            skip_weight_decay_list = self.model.no_weight_decay()
            num_layers = self.model.get_num_layers()       
        if self.cf.layer_decay < 1.0:
            assigner = LayerDecayValueAssigner(list(self.cf.layer_decay ** (num_layers + 1 - i) for i in range(num_layers + 2)))
        else:
            assigner = None
        if self.cf.if_DDP:
            self.optimizer = create_optimizer(
                self.cf, self.model_without_ddp, skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None, 
                get_layer_scale=assigner.get_scale if assigner is not None else None)
        else:
            self.optimizer = create_optimizer(
                self.cf, self.model, skip_list=skip_weight_decay_list,
                get_num_layer=assigner.get_layer_id if assigner is not None else None, 
                get_layer_scale=assigner.get_scale if assigner is not None else None)


    def load_loader(self):
            
        train_dataset, val_dataset, test_dataset, ch_list,test_ch_list = load_dataset(self.cf,self.args, self.logger)

        if self.cf.if_DDP:
            world_size = misc.get_world_size()
            global_rank = misc.get_rank()
            train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank, shuffle=True)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cf.batch_size, sampler=train_sampler,num_workers=self.cf.num_workers,drop_last=True)
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.cf.batch_size, sampler=val_sampler,num_workers=self.cf.num_workers,drop_last=True)
            test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cf.batch_size, sampler=test_sampler,num_workers=self.cf.num_workers,drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.cf.batch_size,drop_last=True, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.cf.batch_size,drop_last=True, shuffle=False)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.cf.batch_size,drop_last=True, shuffle=False)

        return train_loader,val_loader,test_loader, ch_list,test_ch_list
            

    @torch.no_grad()
    def evaluate(self, data_loader,ch_list, header):
        metric_logger = misc.MetricLogger(self.logger,delimiter="  ")
        header = header

        if self.cf.if_DDP:
            self.model_without_ddp.vqgan.quantize.reset_cluster_size(self.cf.device)
        else:
            self.model.vqgan.quantize.reset_cluster_size(self.cf.device)
        if self.logger is not None:
            self.logger.info("Reset the codebook statistic info in quantizer before each epoch")
        
    
        # switch to evaluation mode
        self.model.eval()
        preds = []
        labels = []
        for data,label in metric_logger.log_every(data_loader, 10, header):

            data = data/100 
            data = data.type(torch.FloatTensor).to(self.cf.device) #B,C,2000
            label = torch.Tensor(label).long().to(self.cf.device)
            data = rearrange(data, 'b c (n t) -> b (n c) t ', t=self.cf.sample_rate) #B nc 200  
            
            # compute output
            with torch.cuda.amp.autocast():
                cls_logits, proj_weights = self.model(data, ch_list)
                loss = self.criterion(cls_logits, label)
            
            metric_logger.update(loss=loss.item())
            label = label.detach().cpu().numpy()
            if self.cf.n_class == 2:
                pred = cls_logits.softmax(dim=-1)
                pred = pred[:,1].detach().cpu().numpy()
                if self.cf.if_change_th:
                    if header == 'eval': 
                        threshold = np.sort(pred)[-int(np.sum(label))]
                        print(f'eval threshold,{threshold}')
                    elif header == 'test':
                        threshold = self.threshold
                        print(f'test threshold,{self.threshold}')
                else:
                    threshold = 0.5
                metric_logger.update(threshold=torch.tensor(threshold,device=data.device))
                if sum(label) * (len(label) - sum(label)) != 0:  # to prevent all 0 or all 1 and raise the AUROC error
                    result = binary_metrics_fn(label,pred,metrics=self.metrics,threshold=threshold,)

                else:
                    result = {}
                    for tt in self.metrics:
                        result[tt]=0.0

            else:
                pred = cls_logits.softmax(dim=-1)
                pred = pred.detach().cpu().numpy()
                result = multiclass_metrics_fn(label,pred,metrics=self.metrics,n_labels=list(range(self.cf.n_class)))
            
            for key, value in result.items():
                metric_logger.meters[key].update(value, n=data.shape[0])

        try:
            if self.cf.if_DDP:
                codebook_cluster_size = self.model_without_ddp.vqgan.quantize._codebook.cluster_size
            else:
                codebook_cluster_size = self.model.vqgan.quantize._codebook.cluster_size
        except:
            if self.cf.if_DDP:
                codebook_cluster_size = self.model_without_ddp.vqgan.quantize.cluster_size
            else:
                codebook_cluster_size = self.model.vqgan.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()
            
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
            .format(top1=metric_logger.balanced_accuracy, losses=metric_logger.loss))
        stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

        stat['unused_code']=zero_cnt
        if self.cf.n_class == 2:
            if header == 'eval':
                self.threshold = stat['threshold']
        return stat
   
    def train(self):

        train_loader,val_loader,test_loader, ch_list,test_ch_list = self.load_loader()
        
        # load model
        if self.cf.load_model_path != '':
            checkpoint = torch.load(self.cf.load_model_path, map_location='cpu')
            self.pre_cf = checkpoint['cf']
            self.pre_cf.if_finetune = self.cf.if_finetune
            self.pre_cf.if_scratch = self.cf.if_scratch
            self.pre_cf.n_class = self.cf.n_class
            self.pre_cf.vqgan_model_path = self.cf.vqgan_model_path  #optinal. If you need to use the pre-trained stage2 model, remember to add this command and change the cf.vqgan_model_path to the path of base_class_quantization.pth
            if 'layer_scale_init_values' not in self.pre_cf:
                self.pre_cf.layer_scale_init_values = self.cf.layer_scale_init_values
            if 'drop_path_rate' not in self.pre_cf:
                self.pre_cf.drop_path_rate = self.cf.drop_path_rate
        else:
            tmp_config = Config(self.cf.pretrain_config_path)
            self.pre_cf = tmp_config.fix
            self.pre_cf.if_finetune = self.cf.if_finetune
            self.pre_cf.if_scratch = self.cf.if_scratch
            self.pre_cf.n_class = self.cf.n_class


        self.model = Gram(self.pre_cf).to(device=self.cf.device)
        #
        self.model_ema = None
        if self.cf.if_model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEma(
                self.model,
                decay=self.cf.model_ema_decay,
                device='cpu' if self.cf.model_ema_force_cpu else '',
                resume='')
            print("Using EMA with decay = %.8f" % self.cf.model_ema_decay)
        
        if not self.cf.if_scratch:
            if self.logger != None:
                self.logger.info("Load pre-trained checkpoint from: %s" % self.cf.load_model_path)
            state_dict = self.model.state_dict()
            for k,v in state_dict.items():
                if k in checkpoint['model'] and 'vqgan' not in k:
                    state_dict[k] = checkpoint['model'][k]
            self.model.load_state_dict(state_dict)            
        else:
            if self.logger != None:
                self.logger.info('Training from scartch') 
        
        if self.cf.if_DDP:
            self.model = DDP(self.model, device_ids=[self.args.gpu], output_device=self.args.gpu) #,find_unused_parameters=True
            self.model_without_ddp = self.model.module


        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.random.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)  # if you are using  multi-GPU.
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
     
    
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if self.logger !=None:
            self.logger.info('number of params (M): %.2f' % (n_parameters / 1.e6))

        if self.cf.if_DDP:
            eff_batch_size = self.cf.batch_size * misc.get_world_size()

            if self.cf.lr is None:  # only base_lr is specified
                self.cf.lr = self.cf.blr * eff_batch_size / 256
        else:
            self.cf.lr = self.cf.blr
            eff_batch_size = self.cf.batch_size           

    
        self._get_optim()
        self.optimizer.zero_grad()

        if self.logger !=None:
            self.logger.info("base lr: %.2e" % (self.cf.lr * 256 / eff_batch_size))
            self.logger.info("actual lr: %.2e" % self.cf.lr)
            self.logger.info("effective batch size: %d" % eff_batch_size)
            self.logger.info(f'len(train_loader):  {len(train_loader)}')  
            self.logger.info(f'len(val_loader):  {len(val_loader)}')  
            self.logger.info(f'len(test_loader):  {len(test_loader)}')  
            self.logger.info(self.cf)
            
            
        print("Use step level LR scheduler!")
        num_training_steps_per_epoch = len(train_loader)
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

        best_val_stats = None
        best_test_stats=None
        best_index = 0
        best_epoch = 0
        early_stop_counter = 0
        
        start_time = time.time()
        
        for epoch in range(0, self.cf.epochs):
            if self.cf.if_DDP:
                train_loader.sampler.set_epoch(epoch)
            train_stats = self.train_one_epoch(epoch,num_training_steps_per_epoch, train_loader, ch_list)
            val_stats = self.evaluate(val_loader,ch_list,header='eval')
            if test_ch_list is not None:
                test_stats = self.evaluate(test_loader,test_ch_list,header='test')
            else:
                test_stats = self.evaluate(test_loader,ch_list,header='test')
            
            log_stats = {'epoch': epoch,**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()}}
            if self.logger != None:
                self.logger.info(f'epoch: {epoch}, val_stats: {val_stats}, test_stats: {test_stats}')   
                         
            if self.cf.model_path and misc.is_main_process():
                with open(os.path.join(self.cf.model_path, f"log_{self.cfID}.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
                    
            if misc.is_main_process() and self.summaryWriter is not None:
                for k, v in val_stats.items():
                    self.summaryWriter.add_scalar(f'val/{k}', v, epoch)
                for k, v in test_stats.items():
                    self.summaryWriter.add_scalar(f'test/{k}', v, epoch)            
                self.summaryWriter.flush()
                
            if val_stats[self.main_index] >= best_index:
                best_index = val_stats[self.main_index]
                best_val_stats = val_stats
                best_test_stats = test_stats
                best_epoch = epoch
                early_stop_counter = 0
                if self.logger != None:
                    self.logger.info(f'best epoch: {epoch}, val_stats: {best_val_stats}, test_stats: {best_test_stats}')
                
                to_save = {
                'model' : deepcopy(self.model_without_ddp.state_dict()),
                'optimizer': deepcopy(self.optimizer.state_dict()),
                'epoch': best_epoch,
                'scaler': deepcopy(self.loss_scaler.state_dict()),
                'args': deepcopy(self.args),
                'cf': deepcopy(self.cf),
                'val_stats':best_val_stats,
                'test_stats':best_test_stats,
                }
                if self.cf.if_DDP:
                    to_save['model'] = deepcopy(self.model.module.state_dict())
                    # misc.save_model_finetune(self.cf.model_path,to_save, epoch=None, cfID=self.cfID)
                    misc.save_model(self.cf,self.cf.model_path, epoch=None, model=self.model, model_without_ddp=self.model_without_ddp, 
                                    optimizer=self.optimizer, loss_scaler=self.loss_scaler, 
                                    model_ema=self.model_ema,if_last=False, cfID=self.cfID,step=None, to_save=to_save)

            else:
                early_stop_counter += 1
                if early_stop_counter >= self.cf.early_stop_limit:
                    if self.logger is not None:
                        self.logger.critical("Early stopping")
                    sys.exit(0)

        
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if self.logger !=None:
            self.logger.info('Training time {}'.format(total_time_str))
        return val_stats, test_stats

    def train_one_epoch(self, epoch,  num_training_steps_per_epoch, train_loader, ch_list):

        self.model.train()
        metric_logger = misc.MetricLogger(self.logger,delimiter="  ")
        metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('min_lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = 5
        if self.cf.if_DDP:
            global_rank = misc.get_rank()
        else:
            global_rank = 0
        # iter dataloader

        if self.cf.if_DDP:
            self.model_without_ddp.vqgan.quantize.reset_cluster_size(self.cf.device)
        else:
            self.model.vqgan.quantize.reset_cluster_size(self.cf.device)
        if self.logger is not None:
            self.logger.info("Reset the codebook statistic info in quantizer before each epoch")
        
        
        count = -1
        all_len = len(train_loader)
        for data_iter_step, (data, label) in enumerate(metric_logger.log_every(train_loader, print_freq,global_rank, header)):
            # we use a per iteration (instead of per epoch) lr scheduler
            print(self.cf.run_escription,' ',self.args.seed)
            step = data_iter_step
            if step >= num_training_steps_per_epoch:
                continue
            it = epoch*num_training_steps_per_epoch + data_iter_step  # global training iteration
            # Update LR & WD for the first acc
            if self.lr_schedule_values is not None or self.wd_schedule_values is not None:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    if self.lr_schedule_values is not None:
                        param_group["lr"] = self.lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                    if self.wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = self.wd_schedule_values[it]
            
            count += 1
            if count % self.cf.accum_iter == 0:
                lr_sched.adjust_learning_rate(self.optimizer, count / all_len + epoch, self.cf)
            

            data = data/100
            data = data.type(torch.FloatTensor).to(self.cf.device) #B,C,2000
            data = rearrange(data, 'b c (n t) -> b (n c) t ', t=self.cf.sample_rate) #B nc 200  
            label = torch.Tensor(label).long().to(self.cf.device)
            with torch.cuda.amp.autocast():
                cls_logits, proj_weights = self.model(data, ch_list)
            loss_cls = self.criterion(cls_logits, label)
            loss = loss_cls



            loss_value = loss.item()
            if not math.isfinite(loss_value):
                try:
                    raise Exception()
                except:
                    if self.logger is not None:
                        self.logger.critical("Loss is {}, cls loss {},  stopping training".format(loss_value, loss_cls.item()))
                    # gather the stats from all processes
                    metric_logger.synchronize_between_processes()
                    train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
                    if self.logger is not None:
                        self.logger.info(f"epoch {epoch}, Averaged stats: {str(metric_logger)}")
                    return train_stat


            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(self.optimizer, 'is_second_order') and self.optimizer.is_second_order
            loss /= self.cf.accum_iter
            grad_norm = self.loss_scaler(loss, self.optimizer, clip_grad=self.cf.grad_clip,
                                    parameters=self.model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % self.cf.accum_iter == 0)
            if (data_iter_step + 1) % self.cf.accum_iter == 0:
                self.optimizer.zero_grad()
                if self.model_ema is not None:
                    self.model_ema.update(self.model)
            loss_scale_value = self.loss_scaler.state_dict()["scale"]


            torch.cuda.synchronize()

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_cls=loss_cls.item())
            metric_logger.update(loss=loss_value)
            metric_logger.update(grad_norm=grad_norm)
            metric_logger.update(loss_scale=loss_scale_value)
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
            loss_cls_value_reduce = misc.all_reduce_mean(loss_cls.item())

            if self.summaryWriter is not None and (count + 1) % self.cf.accum_iter == 0:
                """ We use epoch_1000x as the x-axis in tensorboard.
                This calibrates different curves when batch size changes.
                """
                epoch_1000x = int((count / all_len + epoch) * 1000)
                self.summaryWriter.add_scalar('train/train_loss', loss_value_reduce, epoch_1000x)
                self.summaryWriter.add_scalar('train/max_lr', max_lr, epoch_1000x)
                self.summaryWriter.add_scalar('train/min_lr', min_lr, epoch_1000x)
                self.summaryWriter.add_scalar('train/cls_loss', loss_cls_value_reduce, epoch_1000x)


        try:
            if self.cf.if_DDP:
                codebook_cluster_size = self.model_without_ddp.vqgan.quantize._codebook.cluster_size
            else:
                codebook_cluster_size = self.model.vqgan.quantize._codebook.cluster_size
        except:
            if self.cf.if_DDP:
                codebook_cluster_size = self.model_without_ddp.vqgan.quantize.cluster_size
            else:
                codebook_cluster_size = self.model.vqgan.quantize.cluster_size
        zero_cnt = (codebook_cluster_size == 0).sum().item()

        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        train_stat = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        if self.logger is not None:
            self.logger.info(f"epoch {epoch}, Averaged stats: {str(metric_logger)},Unused code in codebook: {zero_cnt}")
        train_stat['unused_code']=zero_cnt
        return train_stat

