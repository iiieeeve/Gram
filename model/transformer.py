"""
taken from: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import collections
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.layers import drop_path

from model.helper import get_1d_sincos_pos_embed_from_grid
from model.encoder import TConv


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self) -> str:
        return 'p={}'.format(self.drop_prob)
    
class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)



class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config, n_embd, n_head):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence

        self.n_head = n_head
        self.if_causal_attention = config.if_causal_attention

    def forward(self, x, ch_list):
        B, T, Emb = x.size()
        
        with torch.no_grad():
            mask = torch.tril(torch.ones(T,T))

        mask = mask.view(1, 1, T, T).to(x.device)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, Emb // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, Emb // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, Emb // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        alpha = 32
        qk = torch.matmul(q/(alpha*math.sqrt(k.size(-1))),k.transpose(-2, -1))
        qk = qk-torch.max(qk,dim=-1,keepdim=True)[0]  #head-wise max

        if self.if_causal_attention:
            qk = qk.masked_fill(mask == 0, float('-inf'))

        att = F.softmax(qk*alpha,dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, Emb)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y  # TODO: check that this does not break anything



class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config, n_embd, n_head, dpr):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(config,n_embd, n_head)
        self.mlp = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(n_embd, 4 * n_embd)),
          ('gelu', nn.GELU()),
          ('fc2', nn.Linear(4 * n_embd, n_embd)),
          ('dp', nn.Dropout(config.resid_pdrop))
        ]))

        if config.if_sandwich_norm:
            self.third_layernorm = nn.LayerNorm(n_embd)
            self.fourth_layernorm = nn.LayerNorm(n_embd)
        self.config=config
        self.drop_path = DropPath(dpr) if dpr > 0. else nn.Identity()
        # self.drop_path = nn.Identity()
        if config.layer_scale_init_values > 0:
            self.gamma_1 = nn.Parameter(config.layer_scale_init_values * torch.ones((n_embd)),requires_grad=True)
            self.gamma_2 = nn.Parameter(config.layer_scale_init_values * torch.ones((n_embd)),requires_grad=True)
        else:
            self.gamma_1, self.gamma_2 = None, None

    def forward(self, x, ch_list, layer_past=None, return_present=False):
        # TODO: check that training still works
    
        if return_present:
            assert not self.training
        # layer past: tuple of length two with B, nh, T, hs
        if self.gamma_1 is None:
            attn = self.drop_path(self.attn(self.ln1(x), ch_list))
        else:
            attn = self.drop_path(self.gamma_1 * self.attn(self.ln1(x), ch_list))
        if self.config.if_sandwich_norm:
            attn = self.third_layernorm(attn)
        x = x + attn
        
        if self.gamma_2 is None:
            fn = self.drop_path(self.mlp(self.ln2(x)))
        else:
            fn = self.drop_path(self.gamma_2 * self.mlp(self.ln2(x)))
        if self.config.if_sandwich_norm:
            fn = self.fourth_layernorm(fn)
        x = x + fn
        if layer_past is not None or return_present:
            return x
        return x


