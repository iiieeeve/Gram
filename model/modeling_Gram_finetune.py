
# --------------------------------------------------------
# References:
# BEiT_v2:https://github.com/Eremurus/BEiTv2
# multi-level feature fusion (MFF): https://github.com/open-mmlab/mmpretrain
# MAE: https://github.com/facebookresearch/mae
# VQGAN: https://github.com/Westlake-AI/VQGAN
# --------------------------------------------------------

from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from model.transformer import Block, trunc_normal_, LabelSmoothingCrossEntropy, TConv
from model.helper import get_1d_sincos_pos_embed_from_grid
from model.modeling_vqkd import VQKD


def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

class Gram(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, cf):
        super().__init__()

        self.cf = cf

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.tmpc=0
        self.tconv = TConv(in_chans=cf.n_chans, out_chans=cf.out_chans)

        self.ch_emb = nn.Parameter(torch.zeros(1, 130, cf.n_embd), requires_grad=False)  # fix sin cos
        self.time_emb = nn.Parameter(torch.zeros(1, 16, cf.n_embd), requires_grad=False)  # fix sin cos
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cf.n_embd))
        self.drop = nn.Dropout(cf.embd_pdrop)

        dpr = [x.item() for x in torch.linspace(0, cf.drop_path_rate, cf.n_layer)]
        self.blocks = nn.ModuleList([Block(cf,cf.n_embd, cf.n_head,dpr[tmp_id]) for tmp_id in range(cf.n_layer)])

        self.norm = nn.LayerNorm(cf.n_embd)
        self.cls_head = nn.Linear(cf.n_embd, cf.n_class)
        
        # MFF
        self.out_indices = cf.out_indices
        if cf.if_mff:
            proj_layers = [
            nn.Linear(cf.n_embd, cf.n_embd)
            for _ in range(len(self.out_indices) - 1)
            ]
            self.proj_layers = torch.nn.ModuleList(proj_layers)
            self.proj_weights = torch.nn.Parameter(
                torch.ones(len(self.out_indices)).view(-1, 1, 1, 1))
            if len(self.out_indices) == 1:
                self.proj_weights.requires_grad = False
        
        # Mimic, only for visible tokens
        

        # if cf.if_mimic:
        #     # self.mimic_head = nn.Linear(cf.n_embd, cf.n_embd)
        #             # task layer
        #     self.mimic_head = nn.Sequential(
        #         nn.Linear(cf.n_embd, cf.n_embd),
        #         nn.Tanh(),
        #         nn.Linear(cf.n_embd, cf.vocab_emb) # for quantize
        #     )

        # --------------------------------------------------------------------------

        # init 
        self.init_std=0.02
        trunc_normal_(self.cls_token, std=self.init_std)
        
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.init_pos_emd()

        self.vqgan = self.load_vqgan(cf)
        for n, p in self.vqgan.named_parameters():
            p.requires_grad=False


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))
        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)


    def init_pos_emd(self):
        ch_embed = get_1d_sincos_pos_embed_from_grid(self.ch_emb.shape[-1], self.ch_emb.shape[-2])  # (C,emb)
        self.ch_emb.data.copy_(torch.from_numpy(ch_embed).float().unsqueeze(0))
        time_emb = get_1d_sincos_pos_embed_from_grid(self.time_emb.shape[-1], self.time_emb.shape[-2])  # (C,emb)
        self.time_emb.data.copy_(torch.from_numpy(time_emb).float().unsqueeze(0))
    

    @staticmethod
    def load_vqgan(cf):
        # print(cf.gpu)
        checkpoint = torch.load(cf.vqgan_model_path, map_location='cpu')
        pre_cf = checkpoint['cf']
        if 'perceptive_loss' in cf.vqgan_model_path:
            pre_cf.if_perceptual_loss = True
        else:
            pre_cf.if_perceptual_loss = False
        model = VQKD(pre_cf)
        model.load_state_dict(checkpoint['model'])
        model = model.eval()
        return model
    
    @torch.no_grad()
    def encode_to_z(self, x, ch_list):
        # target = self.vqgan.encoder(x, ch_list)
        quantize, embed_ind, _= self.vqgan.encode(x, ch_list)  #quantize: B nc vocab_emb
        indices = embed_ind.view(quantize.shape[0], -1)  #b, nc
        target = self.vqgan.to_quantizer_features
        return target, quantize, indices  

    
    @torch.no_grad()
    def z_to_image(self, indices, ch_list):
        bsz, seq_len = indices.size()
        quantize = self.vqgan.quantize.embedding(indices).view((bsz, seq_len,self.cf.vocab_emb))
        x_rec = self.vqgan.decode(quantize, ch_list)
        return x_rec
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'tok_emb', 'cls_token', 'mask_token'}

    def get_num_layers(self):
        return len(self.blocks)
        
    def get_block_size(self):
        return self.block_size


    def get_less_channel_emb(self,device,ch_list):
        new_ch_list = [i.upper() for i in ch_list]
        try:
            ids_keep = [self.cf.all_ch_list.index(i) for i in new_ch_list]
            ids_keep = torch.tensor(ids_keep, device=device)
            ids_keep = ids_keep[None,:,None] #1 C 1
            return ids_keep
        except:
            print('new_ch_list does not match ori_ch_list')

    def forward_encoder(self, x, ch_list):
        # embed patches
        
        b,nc,e=x.shape
        assert nc%len(ch_list) == 0 
        n=int(nc/len(ch_list))
        
        # add pos embed w/o cls token

        if len(ch_list)!=130:
            channel_embed_id = self.get_less_channel_emb(x.device,ch_list)  #1 C 1
            channel_embed_ = torch.gather(self.ch_emb, dim=1, index=channel_embed_id.expand(-1, -1, self.ch_emb.shape[-1]))  #1 C emb
            channel_embed_ = torch.as_tensor(channel_embed_, device=x.device).unsqueeze(1).expand(b,n,-1,-1).flatten(1, 2)#b n c emb
            #B 18 T_v        
        else:
            channel_embed_ = torch.as_tensor(self.ch_emb, device=x.device).unsqueeze(1).expand(b,n,-1,-1).flatten(1, 2)#b nc emb
        channel_embed_.requires_grad_(False)
        assert n <= 16, "Cannot forward, model block size is exhausted."

        time_embeddings = self.time_emb[:, 0:n, :].unsqueeze(2).expand(b, -1, len(ch_list), -1).flatten(1, 2)
        emb = time_embeddings+channel_embed_
   
        x = self.tconv(x)
        x = x + emb

        # append cls token
        cls_tokens = self.cls_token.expand(b, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)

        res = []
        # apply Transformer blocks
        if self.cf.if_mff:
            # self.tmpc =self.tmpc+1
            for i, blk in enumerate(self.blocks):
                x = blk(x, ch_list)
                if i in self.out_indices:
                    if i != self.out_indices[-1]:
                        proj_x = self.proj_layers[self.out_indices.index(i)](x)
                    else:
                        proj_x = x
                    res.append(proj_x)
            res = torch.stack(res)
            proj_weights = F.softmax(self.proj_weights, dim=0)
            res = res * proj_weights
            res = res.sum(dim=0)
            x = self.norm(res)
        else:
            proj_weights = torch.ones(len(self.out_indices)).view(-1, 1, 1, 1)
            for i, blk in enumerate(self.blocks):
                x = blk(x, ch_list)
            x = self.norm(x)
        # use cls token
        x = x[:, 0, :]
        # predictor projection
        x = self.cls_head(x)
            
        return x, proj_weights.view(-1)
    
    def forward(self, x, ch_list):

        latent,  proj_weights= self.forward_encoder(x, ch_list)
    

        return latent, proj_weights
    
if __name__=='__main__':
    from utils.config import Config, get_param_sets
    config = Config('./config/seed5_finetune.yaml')
    cf_tuning=config.tuning
    cf_fix = config.fix
    cfs = get_param_sets(cf_tuning)
    cf = cfs[0]
    cf.update(cf_fix)
    
    checkpoint = torch.load('./checkpoints/base.pth', map_location='cpu')  
    pre_cf = checkpoint['cf']
    pre_cf.if_finetune = cf.if_finetune
    pre_cf.if_scratch = cf.if_scratch
    pre_cf.n_class = cf.n_class   
    pre_cf.vqgan_model_path = cf.vqgan_model_path  #optinal. If you need to use the pre-trained stage2 model, remember to add this command and change the cf.vqgan_model_path to the path of base_class_quantization.pth
    if 'layer_scale_init_values' not in pre_cf: 
        pre_cf.layer_scale_init_values = cf.layer_scale_init_values
    if 'drop_path_rate' not in pre_cf:
        pre_cf.drop_path_rate = cf.drop_path_rate

    model = Gram(pre_cf).to(device=cf.device)
    state_dict = model.state_dict()
    for k,v in state_dict.items():
        if k in checkpoint['model'] and 'vqgan' not in k:
            state_dict[k] = checkpoint['model'][k]
    model.load_state_dict(state_dict)   

    data = torch.rand(1,17,200).to(device=cf.device)
    ch_list =  ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8','T7', 'T8', 'P7', 'P8', 'CZ']
    cls_logits, proj_weights = model(data, ch_list)
    print(cls_logits, proj_weights)