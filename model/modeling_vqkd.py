
# --------------------------------------------------------
# References:
# BEiT_v2:https://github.com/Eremurus/BEiTv2
# multi-level feature fusion (MFF): https://github.com/open-mmlab/mmpretrain
# MAE: https://github.com/facebookresearch/mae
# VQGAN: https://github.com/Westlake-AI/VQGAN
# --------------------------------------------------------

import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import partial, reduce
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model

from model.encoder import Encoder
from model.decoder import Decoder
from model.norm_ema_quantizer import NormEMAVectorQuantizer

class GetFrequencyFea(nn.Module):
    def __init__(self,stft_n=64,hop_length=16, input_T=200):
        super().__init__()
        # self.projection = nn.Linear(n_freq, emb_size)
        self.stft_n = stft_n
        self.hop_length = hop_length
        self.stft_size = self._get_size(input_T)

    def stft(self, sample):
        #sample #B,C,T
        signal = []
        for s in range(sample.shape[1]):
            spectral = torch.stft(
                sample[:, s, :],  
                n_fft=self.stft_n,
                hop_length=self.hop_length,
                normalized=False,
                center=False,
                onesided=True,
                return_complex=True,
            )
            spectral = rearrange(spectral,'b f t -> b (t f)')
            signal.append(spectral) 
        stacked = torch.stack(signal).permute(1,0,2)  #nc b tf -> b nc tf
        
        return torch.abs(stacked)  #b nc tf
       
    def _get_size(self,input_T):
        x = torch.ones(1,1,input_T)
        stacked = self.stft(x) #b nc tf
        return stacked.shape[2]

    def forward(self, x):
        """
        x: (batch,channel, time)
        out: (batch, time, emb_size)

        method 'stft' 
        """
        with torch.no_grad():
            stacked = self.stft(x)  #b,t,f,nc
        # x = self.projection(stacked)  #b,t,emb
        return stacked

class VQKD(nn.Module):
    def __init__(self,cf,**kwargs ):
        super().__init__()
        print(kwargs)
        self.cf = cf
        # encoder & decode params
        self.encoder = Encoder(cf)
        self.decoder = Decoder(cf)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=cf.vocab_size, embedding_dim=cf.vocab_emb, beta=1, kmeans_init=cf.quantize_kmeans_init, decay=0.99,
        )
        
        # reg target
        self.regress_target = cf.regress_target
        if self.regress_target == 'raw':
            self.decoder_out_dim = cf.model_window_size
        else:   
            raise NotImplementedError

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(cf.n_embd, cf.n_embd),
            nn.Tanh(),
            nn.Linear(cf.n_embd, cf.vocab_emb) # for quantize
        )
        self.decode_task_layer = nn.Identity()
        
        self.rec_loss_type = cf.rec_loss_type
        self.mse_loss = F.mse_loss

        self.logit_laplace_eps = 0.1
        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.ch_emb', 'decoder.time_emb', 
                'encoder.ch_emb', 'encoder.time_emb'}
    

    @property
    def device(self):
        return self.decoder.cls_token.device


    def encode(self, x, ch_list):
        encoder_features = self.encoder(x, ch_list)  #b nc emb

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))  #b nc vocab_emb

        self.to_quantizer_features=to_quantizer_features

        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss
    
    def decode(self, quantize, ch_list, **kwargs):
        decoder_features = self.decoder(quantize, ch_list)
        rec = self.decode_task_layer(decoder_features)

        return rec

    @torch.no_grad()
    def get_regress_target(self, x, **kwargs):       
        target = x
        return target

    def std_norm(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x - mean) / (std+ 1.e-5)
        return x

    def calculate_rec_loss(self, rec, target):  
        if self.rec_loss_type == 'cosine':
            target = target / target.norm(dim=-1, keepdim=True)
            rec = rec / rec.norm(dim=-1, keepdim=True)
            rec_loss = (1 - (target * rec).sum(-1)).mean()
        elif self.rec_loss_type == 'mse':  
            target = self.std_norm(target)
            rec = self.std_norm(rec)
            rec_loss = self.mse_loss(rec, target)  
        else:
            raise NotImplementedError

        return rec_loss

    
    def forward(self, x, ch_list, dis_fake=None):

        target = self.get_regress_target(x)
        
        quantize, embed_ind, emb_loss = self.encode(x, ch_list)
        xrec = self.decode(quantize, ch_list)

        rec_loss = self.calculate_rec_loss(xrec, target)
        loss = emb_loss + rec_loss

        log = {}

        log[f'quant_loss'] = emb_loss.detach().mean()
        log[f'rec_loss'] = rec_loss.detach().mean()
        log[f'total_loss'] = loss.detach().mean()

        return xrec,loss, log, embed_ind




if __name__ =='__main__':
    from utils.config import Config, get_param_sets
    config = Config('./config/pretrain_base_class_quantization.yaml')
    cf_tuning=config.tuning
    cf_fix = config.fix
    cfs = get_param_sets(cf_tuning)
    cf = cfs[0]
    cf.update(cf_fix)
    model = VQKD(cf).to(device=cf.device)
    checkpoint = torch.load('./checkpoints/base_class_quantization.pth', map_location='cpu')  
    model.load_state_dict(checkpoint['model'])  
    data = torch.rand(1,17,200).to(device=cf.device)
    ch_list =  ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8','T7', 'T8', 'P7', 'P8', 'CZ']
    data_rec, loss, loss_log, indices = model(data, ch_list)
    print(data_rec, loss, loss_log, indices)

