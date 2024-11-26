import h5py
import numpy as np
from pathlib import Path

"""
HMC: 1139.80 h,  ch_num 4, fsize 24.95 G
TUEV_train: 115.07 h,  ch_num 19, fsize 12.09 G
TUEV_eval: 40.86 h,  ch_num 19, fsize 4.29 G
seed-5: 51.43 h,  ch_num 62, fsize 17.11 G
shu_dataset: 13.32 h,  ch_num 32, fsize 2.34 G
MODMA: 4.46 h,  ch_num 19, fsize 0.46 G
EH: 213.41 h,  ch_num 18, fsize 20.69 G

"""

hdf5_path = Path('your data path')

ch_num=None
all_time = 0
for h5_file_path in hdf5_path.iterdir():
    h5_file_name = h5_file_path.stem
    if 'EH' in h5_file_name:
        h5_file = h5py.File(h5_file_path,'r')
        print((f'preprocessing {h5_file_name}'))
        fsize=h5_file_path.stat().st_size
        fsize=fsize/(1024**3)
        
        count = 0
        for group_name in h5_file.keys():
            for da_name in h5_file[group_name].keys():
                if 'label' in da_name:
                    continue
                da = h5_file[group_name][da_name]
                sample_rate = da.attrs[ 'rsFreq']
                tim = da.shape[1]
                if ch_num==None:
                    ch_num = da.shape[0]
                if ch_num != da.shape[0]:
                    raise Exception(f'{h5_file_name} {group_name} {da_name} ch_num: {da.shape[0]}, others_ch_num: {ch_num}')
                count += tim/sample_rate
        print('{}: {:.2f} h,  ch_num {}, fsize {:.2f} G'.format(h5_file_name, count/3600,ch_num,  fsize))
        all_time += count/3600
print(f'all time {all_time}')
                
            
        