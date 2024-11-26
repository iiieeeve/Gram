
# --------------------------------------------------------
# References:
# BEiT_v2:https://github.com/Eremurus/BEiTv2
# multi-level feature fusion (MFF): https://github.com/open-mmlab/mmpretrain
# MAE: https://github.com/facebookresearch/mae
# VQGAN: https://github.com/Westlake-AI/VQGAN
# --------------------------------------------------------

# print(Path(__file__).parent.parent)
from functools import partial

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from collections import OrderedDict


from model.transformer import Block, trunc_normal_, LabelSmoothingCrossEntropy, TConv
from model.helper import get_1d_sincos_pos_embed_from_grid
from model.modeling_vqkd import VQKD



def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

class Gram(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, args):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.args = args
        self.tconv = TConv(in_chans=args.n_chans, out_chans=args.out_chans)

        self.ch_emb = nn.Parameter(torch.zeros(1, 130, args.n_embd), requires_grad=False)  # fix sin cos
        self.time_emb = nn.Parameter(torch.zeros(1, 16, args.n_embd), requires_grad=False)  # fix sin cos
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.n_embd))
        self.drop = nn.Dropout(args.embd_pdrop)

        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, args.n_layer)]
        self.blocks = nn.ModuleList([Block(args,args.n_embd, args.n_head,dpr[tmp_id]) for tmp_id in range(args.n_layer)])

        self.norm = nn.LayerNorm(args.n_embd)
        
        # MFF
        self.out_indices = args.out_indices
        if args.if_mff:
            proj_layers = [
            nn.Linear(args.n_embd, args.n_embd)
            for _ in range(len(self.out_indices) - 1)
            ]
            self.proj_layers = torch.nn.ModuleList(proj_layers)
            self.proj_weights = torch.nn.Parameter(
                torch.ones(len(self.out_indices)).view(-1, 1, 1, 1))
            if len(self.out_indices) == 1:
                self.proj_weights.requires_grad = False
    
    
        if args.if_mimic:
            self.mimic_head = nn.Linear(args.n_embd, args.target_n_embd)

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(args.n_embd, args.decoder_n_embd, bias=True)
        if not self.args.if_pad_with_cls_token:
            self.mask_token = nn.Parameter(torch.zeros(1, 1, args.decoder_n_embd))
        self.decoder_ch_emb = nn.Parameter(torch.zeros(1, 130, args.decoder_n_embd), requires_grad=False)  # fix sin cos
        self.decoder_time_emb = nn.Parameter(torch.zeros(1, 16, args.decoder_n_embd), requires_grad=False)  # fix sin cos
        
        decoder_dpr = [x.item() for x in torch.linspace(0, args.decoder_drop_path_rate, args.decoder_n_layer)]
        self.decoder_blocks = nn.ModuleList([Block(args,args.decoder_n_embd, args.decoder_n_head,decoder_dpr[tmp_id]) for tmp_id in range(args.decoder_n_layer)])


        self.decoder_norm = nn.LayerNorm(args.decoder_n_embd)
        self.decoder_pred = nn.Linear(args.decoder_n_embd, args.vocab_size, bias=False)  # decoder to patch
        self.criterion = nn.CrossEntropyLoss()
        # init 
        self.init_std=0.02
        trunc_normal_(self.cls_token, std=self.init_std)
        if not self.args.if_pad_with_cls_token:
             trunc_normal_(self.mask_token, std=self.init_std)
        
        self.apply(self._init_weights)
        self.fix_init_weight()
        self.init_pos_emd()

        self.vqgan = self.load_vqgan(args)
        for n, p in self.vqgan.named_parameters():
            p.requires_grad=False
        if self.args.mimic_loss_type == 'mse':  
            self.mse_loss = F.mse_loss


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
        for layer_id, layer in enumerate(self.decoder_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_pos_emd(self):
        ch_embed = get_1d_sincos_pos_embed_from_grid(self.ch_emb.shape[-1], self.ch_emb.shape[-2])  # (C,emb)
        self.ch_emb.data.copy_(torch.from_numpy(ch_embed).float().unsqueeze(0))
        time_emb = get_1d_sincos_pos_embed_from_grid(self.time_emb.shape[-1], self.time_emb.shape[-2])  # (C,emb)
        self.time_emb.data.copy_(torch.from_numpy(time_emb).float().unsqueeze(0))
        decoder_ch_embed = get_1d_sincos_pos_embed_from_grid(self.decoder_ch_emb.shape[-1], self.decoder_ch_emb.shape[-2])  # (C,emb)
        self.decoder_ch_emb.data.copy_(torch.from_numpy(decoder_ch_embed).float().unsqueeze(0))
        decoder_time_emb = get_1d_sincos_pos_embed_from_grid(self.decoder_time_emb.shape[-1], self.decoder_time_emb.shape[-2])  # (C,emb)
        self.decoder_time_emb.data.copy_(torch.from_numpy(decoder_time_emb).float().unsqueeze(0))
    

    @staticmethod
    def load_vqgan(args):
        # print(args.gpu)
        checkpoint = torch.load(args.vqgan_model_path, map_location='cpu')
        pre_cf = checkpoint['cf']
        if 'perceptive_loss' in args.vqgan_model_path:
            pre_cf.if_perceptual_loss = True
        else:
            pre_cf.if_perceptual_loss = False
        model = VQKD(pre_cf)
        model.load_state_dict(checkpoint['model'])
        model = model.eval()
        return model
    
    
    @torch.no_grad()
    def encode_to_z(self, x, ch_list):
        target = self.vqgan.encoder(x, ch_list)
        quantize, embed_ind, _= self.vqgan.encode(x, ch_list)  #quantize: B nc vocab_emb
        indices = embed_ind.view(quantize.shape[0], -1)  #b, nc
        return target, quantize, indices  
    
    @torch.no_grad()
    def z_to_image(self, indices, ch_list):
        bsz, seq_len = indices.size()
        quantize = self.vqgan.quantize.embedding(indices).view((bsz, seq_len,self.args.vocab_emb))
        x_rec = self.vqgan.decode(quantize, ch_list)
        return x_rec
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'tok_emb', 'cls_token', 'mask_token'}

    def get_num_layers(self):
        return len(self.blocks)
        
    def get_block_size(self):
        return self.block_size


    def random_masking(self, x, emb, mask_ratio, input_noise=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        if input_noise is None:
            noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        else:
            noise = input_noise
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]


        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask, ids_restore

    def get_less_channel_emb(self,device,ch_list):
        new_ch_list = [i.upper() for i in ch_list]
        try:
            ids_keep = [self.args.all_ch_list.index(i) for i in new_ch_list]
            ids_keep = torch.tensor(ids_keep, device=device)
            ids_keep = ids_keep[None,:,None] #1 C 1
            return ids_keep
        except:
            print('new_ch_list does not match ori_ch_list')

    def forward_encoder(self, x, ch_list,mask_ratio=None, input_noise=None):
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

        ids_keep, mask, ids_restore = self.random_masking(x, emb, self.args.mask_ratio)            
        x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        emb = torch.gather(emb, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, emb.shape[2]))

        x = self.tconv(x)
        x = x + emb

        # append cls token
        cls_tokens = self.cls_token.expand(b, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)

        res = []
        # apply Transformer blocks
        if self.args.if_mff:
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
            
        return x, mask,ids_keep, ids_restore, proj_weights.view(-1)

    def forward_decoder(self, x, ch_list, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        if self.args.if_pad_with_cls_token:
            mask_tokens = x[:, 0:1].repeat(1, ids_restore.shape[1] + 1 - x.shape[1], 1)
        else:
            print('using mask token')
            mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)

        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        

        # add pos embed
        b,nc,e=x_.shape
        assert nc%len(ch_list) == 0 
        n=int(nc/len(ch_list))
        
        # add pos embed w/o cls token

        if len(ch_list)!=130:
            channel_embed_id = self.get_less_channel_emb(x_.device,ch_list)  #1 C 1
            channel_embed_ = torch.gather(self.decoder_ch_emb, dim=1, index=channel_embed_id.expand(-1, -1, self.decoder_ch_emb.shape[-1]))  #1 C emb
            channel_embed_ = torch.as_tensor(channel_embed_, device=x_.device).unsqueeze(1).expand(b,n,-1,-1).flatten(1, 2)#b nc emb
            #B 18 T_v        
        else:
            channel_embed_ = torch.as_tensor(self.decoder_ch_emb, device=x_.device).unsqueeze(1).expand(b,n,-1,-1).flatten(1, 2)#b nc emb
        channel_embed_.requires_grad_(False)
        assert n <= 16, "Cannot forward, model block size is exhausted."

        time_embeddings = self.decoder_time_emb[:, 0:n, :].unsqueeze(2).expand(b, -1, len(ch_list), -1).flatten(1, 2)
        x_ = x_ + time_embeddings+channel_embed_


        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, ch_list)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)
        
        # remove cls token
        x = x[:, 1:, :]
        
        return x 

    def cls_loss(self, gt_indices, logits, mask):
        bsz, seq_len = gt_indices.size()
        mask = mask.to(torch.bool)
        gt_indices=gt_indices[mask]
        logits = logits[mask]
        loss = self.criterion(logits, gt_indices)
        
        mlm_acc = (logits.max(-1)[1] == gt_indices).float().mean()
        return loss, mlm_acc
    

    def std_norm(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x - mean) / (std+ 1.e-5)
        return x

    def mimic_loss(self, rec, target, ids_keep):  
        # only for visible tokens
        if self.args.mimic_loss_type == 'cosine':
            target = torch.gather(target, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, target.shape[-1]))
            target = l2norm(target)
            rec = l2norm(rec)
            rec_loss = (1 - (target * rec).sum(-1)).mean()
        elif self.args.mimic_loss_type == 'mse':  
            target = torch.gather(target, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, target.shape[-1]))
            target = self.std_norm(target)
            rec = self.std_norm(rec)
            rec_loss = self.mse_loss(rec, target)

        else:
            raise NotImplementedError

        return rec_loss
    
    def forward(self, x, ch_list,mask_ratio=None, input_noise=None):
        with torch.no_grad():
            _, _, indices = self.encode_to_z(x, ch_list)
        latent, mask, ids_keep, ids_restore, proj_weights= self.forward_encoder(x, ch_list,mask_ratio, input_noise)
        if self.args.if_mimic:
            # only for visible tokens: rec vs target
            rec = self.mimic_head(latent[:,1:,:])
            x_fft = torch.fft.fft(x, dim=-1)
            target = torch.abs(x_fft)
            mimic_loss = self.mimic_loss(rec, target,ids_keep)
        else:
            mimic_loss = torch.tensor(-1.0, device=x.device)
            
        pred = self.forward_decoder(latent, ch_list, ids_restore) 
        cls_loss, mlm_acc = self.cls_loss(indices, pred, mask)
        return cls_loss, mimic_loss, mlm_acc, pred, mask, proj_weights

if __name__ =='__main__':
    from utils.config import Config, get_param_sets
    config = Config('./config/pretrain_base.yaml')
    cf_tuning=config.tuning
    cf_fix = config.fix
    cfs = get_param_sets(cf_tuning)
    cf = cfs[0]
    cf.update(cf_fix)
    model = Gram(cf).to(device=cf.device)
    checkpoint = torch.load('./checkpoints/base.pth', map_location='cpu')  
    model.load_state_dict(checkpoint['model'])  
    data = torch.rand(1,17,200).to(device=cf.device)
    ch_list =  ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8','T7', 'T8', 'P7', 'P8', 'CZ']
    cls_loss, mimic_loss, mlm_acc, pred, mask, proj_weights= model(data, ch_list)
    print(cls_loss, mimic_loss, mlm_acc)