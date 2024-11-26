import torch.nn as nn
import torch
from torch.nn import functional as F

from einops import rearrange
import math
from timm.models.layers import drop_path, to_2tuple, trunc_normal_ 
from model.helper import get_1d_sincos_pos_embed_from_grid
import collections


class TConv(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, in_chans=1, out_chans=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, out_chans, kernel_size=(1, 15), stride=(1, 8), padding=(0, 7))
        self.gelu1 = nn.GELU()
        self.norm1 = nn.GroupNorm(4, out_chans)
        self.conv2 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.gelu2 = nn.GELU()
        self.norm2 = nn.GroupNorm(4, out_chans)
        self.conv3 = nn.Conv2d(out_chans, out_chans, kernel_size=(1, 3), padding=(0, 1))
        self.norm3 = nn.GroupNorm(4, out_chans)
        self.gelu3 = nn.GELU()

    def forward(self, x, **kwargs):
        # x = rearrange(x, 'B N A T -> B (N A) T')
        B, NA, T = x.shape
        x = x.unsqueeze(1)
        x = self.gelu1(self.norm1(self.conv1(x)))
        x = self.gelu2(self.norm2(self.conv2(x)))
        x = self.gelu3(self.norm3(self.conv3(x)))
        x = rearrange(x, 'B C NA T -> B NA (T C)')
        return x



class SelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.n_head = n_head

    def forward(self, x):
        B, T, Emb = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, Emb // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, Emb // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, Emb // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        alpha = 32
        qk = torch.matmul(q/(alpha*math.sqrt(q.size(-1))),k.transpose(-2, -1))
        qk = qk-torch.max(qk,dim=-1,keepdim=True)[0]  #head-wise max

        att = F.softmax(qk*alpha,dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, Emb)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop,if_sandwich_norm):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = SelfAttention(n_embd, n_head, attn_pdrop, resid_pdrop)
        self.mlp = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(n_embd, 4 * n_embd)),
          ('gelu', nn.GELU()),
          ('fc2', nn.Linear(4 * n_embd, n_embd)),
          ('dp', nn.Dropout(resid_pdrop))
        ]))


        if if_sandwich_norm:
            self.third_layernorm = nn.LayerNorm(n_embd)
            self.fourth_layernorm = nn.LayerNorm(n_embd)
        self.if_sandwich_norm=if_sandwich_norm

    def forward(self, x):
        attn = self.attn(self.ln1(x))
        if self.if_sandwich_norm:
            attn = self.third_layernorm(attn)
        x = x + attn
        fn = self.mlp(self.ln2(x))
        if self.if_sandwich_norm:
            fn = self.fourth_layernorm(fn)
        x = x + fn
        return x


class Encoder(nn.Module):
    def __init__(self, args, **kargs):
        super().__init__()
        self.args = args
        self.tconv = TConv()
        self.block_size=args.block_size
        self.ch_emb = nn.Parameter(torch.zeros(1, 130, args.n_embd), requires_grad=False)  # fix sin cos
        self.time_emb = nn.Parameter(torch.zeros(1, 16, args.n_embd)) # learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.n_embd))
        self.drop = nn.Dropout(args.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(args.n_embd, args.encoder_n_head, args.attn_pdrop, args.resid_pdrop,args.if_sandwich_norm) for _ in range(args.encoder_n_layer)])
        self.ln_f = nn.LayerNorm(args.n_embd)
        trunc_normal_(self.time_emb, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        ch_embed = get_1d_sincos_pos_embed_from_grid(self.ch_emb.shape[-1], self.ch_emb.shape[-2])  # (C,emb)
        self.ch_emb.data.copy_(torch.from_numpy(ch_embed).float().unsqueeze(0))

        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def get_less_channel_emb(self,idx,ch_list):
        new_ch_list = [i.upper() for i in ch_list]
        try:
            ids_keep = [self.args.all_ch_list.index(i) for i in new_ch_list]
            ids_keep = torch.tensor(ids_keep, device=idx.device)
            ids_keep = ids_keep[None,:,None] #1 C 1
            return ids_keep
        except:
            print('new_ch_list does not match ori_ch_list')

    def forward(self, x, ch_list):

        b,nc,e=x.shape
        assert nc%len(ch_list) == 0 
        n=int(nc/len(ch_list))

        cls_tokens = self.cls_token.expand(b, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1)

        if len(ch_list)!=130:
            channel_embed_id = self.get_less_channel_emb(x,ch_list)  #1 C 1
            channel_embed_ = torch.gather(self.ch_emb, dim=1, index=channel_embed_id.expand(-1, -1, self.ch_emb.shape[-1]))  #1 C emb
            channel_embed_ = torch.as_tensor(channel_embed_, device=x.device).unsqueeze(1).expand(b,n,-1,-1).flatten(1, 2)#1 n c emb
            #B 18 T_v        
        else:
            channel_embed_ = torch.as_tensor(self.ch_emb, device=x.device).unsqueeze(1).expand(b,n,-1,-1).flatten(1, 2)#1 nc emb
        channel_embed_.requires_grad_(False)
        assert n <= 16, "Cannot forward, model block size is exhausted."

        time_embeddings = self.time_emb[:, 0:n, :].unsqueeze(2).expand(b, -1, len(ch_list), -1).flatten(1, 2)

        x = x[:, 1:, :] #remove cls token
        emb = time_embeddings+channel_embed_  
        
    
        x = self.tconv(x)  #b,nc, emb
        x = x + emb
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return x

