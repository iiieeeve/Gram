import torch.nn as nn
import torch
from torch.nn import functional as F

from einops import rearrange
import math
from timm.models.layers import drop_path, to_2tuple, trunc_normal_ 
from model.helper import get_1d_sincos_pos_embed_from_grid
# from model.modeling_finetune import Block
from model.encoder import Block, TConv


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=2000, patch_size=200, in_chans=1, embed_dim=200):
        super().__init__()
        # img_size = to_2tuple(img_size)
        # patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(1,patch_size), stride=(1, patch_size))

    def forward(self, x, **kwargs):
        B, C, N, CH = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x




class Decoder(nn.Module):
    def __init__(self, args, **kargs):
        super().__init__()
        self.args = args
        self.block_size = args.block_size
        self.conv_emb = PatchEmbed(img_size=args.window_size, patch_size=1, in_chans=args.vocab_emb, embed_dim=args.n_embd)

        self.ch_emb = nn.Parameter(torch.zeros(1, 130, args.n_embd), requires_grad=False)  # fix sin cos
        self.time_emb = nn.Parameter(torch.zeros(1, 16, args.n_embd)) # learnable
        self.cls_token = nn.Parameter(torch.zeros(1, 1, args.n_embd))
        self.drop = nn.Dropout(args.embd_pdrop)
        # transformer
        self.blocks = nn.ModuleList([Block(args.n_embd, args.decoder_n_head, args.attn_pdrop, args.resid_pdrop,args.if_sandwich_norm) for _ in range(args.decoder_n_layer)])

        self.tconv = TConv()
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
        #x: b nc emb
        x = rearrange(x, 'b (n c) e -> b e n c ', c=len(ch_list)).contiguous()
        x = self.conv_emb(x)  # b (nc) e
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
        assert n <= self.block_size, "Cannot forward, model block size is exhausted."

        time_embeddings = self.time_emb[:, 0:n, :].unsqueeze(2).expand(b, -1, len(ch_list), -1).flatten(1, 2)

        x = x[:, 1:, :] + time_embeddings+channel_embed_  #remove cls token

        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.tconv(x)
        x = self.ln_f(x)
        return x
