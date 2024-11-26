
import sys
from pathlib import Path
sys.path.append('/home/zhaoliming/Gram/')
import torch
from model.modeling_Gram import Gram as stage_Gram

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))


    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    # print(f'outside mask {mask}')
    return ids_keep,ids_restore, mask, noise,


if __name__ == '__main__':
    noise_rate = 0.3
    load_gen_model_path = './checkpoints/base.pth'
    checkpoint = torch.load(load_gen_model_path, map_location='cpu')
    tmp_pre_cf = checkpoint['cf']
    if 'target_n_embd' not in tmp_pre_cf:
        tmp_pre_cf['target_n_embd'] = 200
    tmp_pre_cf.vqgan_model_path = './checkpoints/base_class_quantization.pth'

    data = torch.rand(1,17,200).to(device=tmp_pre_cf.device)
    ch_list =  ['FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8','T7', 'T8', 'P7', 'P8', 'CZ']

    ids_keep,ids_restore, mask, noise, = random_masking(data, noise_rate)

    mask = mask.unsqueeze(-1)   #b, nc，1
    tmp = torch.where(mask==1)
    input_noise= [ids_keep, mask, ids_restore]


    stage2_model = stage_Gram(tmp_pre_cf).to(device=tmp_pre_cf.device)
    stage2_model.load_state_dict(checkpoint['model']) 
    stage2_model.eval()
    with torch.no_grad():
        _, _, _, pred, _, _ = stage2_model(data, ch_list, noise_rate, input_noise)
        indices = torch.argmax(pred,dim=-1)

        new_data = stage2_model.z_to_image(indices, ch_list)
        new_data = new_data*10
        final_data = data * (1 - mask) + new_data * mask   #1: mask  0: remain
  
    print(final_data)
