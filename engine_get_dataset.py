import sys
import h5py
from pathlib import Path
import random
import numpy as np
import sys
import os
import torch
import pickle
from scipy.signal import resample
from collections import Counter



def load_dataset(cf,args, logger):
    test_ch_list = None

    if cf.dataset =='TUEV' :
        from shock.dataset.TUEV import TUEVDataset
        tmp = h5py.File(Path(cf.h5_data_path)/('TUEV_train.hdf5'),'r')
        sub_names = list(tmp.keys())
        sub_names = sorted(sub_names, reverse=True)

        train_subs = sub_names[:int(len(sub_names)*cf.train_ratio)]
        val_subs = sub_names[int(len(sub_names)*cf.train_ratio):]

        train_dataset = TUEVDataset(file_path=Path(cf.h5_data_path)/('TUEV_train.hdf5'),sub_names=train_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        val_dataset = TUEVDataset(file_path=Path(cf.h5_data_path)/('TUEV_train.hdf5'),sub_names=val_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        test_dataset = TUEVDataset(file_path=Path(cf.h5_data_path)/('TUEV_eval.hdf5'), window_size=cf.window_size, stride_size=cf.stride_size)
        
        ch_list = cf.TUEV
    
    elif cf.dataset =='seed5' :
        from shock.dataset.seed5 import DownstreamDataset
        
        train_dataset = DownstreamDataset(Path(cf.h5_data_path)/('seed-5.hdf5'),label_map=cf.label_map,window_size=cf.window_size, stride_size=cf.stride_size, subject_start_percentage=0.4, subject_end_percentage=1)
        val_dataset = DownstreamDataset(Path(cf.h5_data_path)/('seed-5.hdf5'),label_map=cf.label_map, window_size=cf.window_size, stride_size=cf.stride_size, subject_start_percentage=0.2, subject_end_percentage=0.4)
        test_dataset = DownstreamDataset(Path(cf.h5_data_path)/('seed-5.hdf5'),label_map=cf.label_map, window_size=cf.window_size, stride_size=cf.stride_size, subject_start_percentage=0, subject_end_percentage=0.2)
    
        ch_list = train_dataset.get_ch_names()

    elif cf.dataset == 'HMC':

        tmp = h5py.File(Path(cf.h5_data_path)/('HMC.hdf5'),'r')
        sub_names = list(tmp.keys())
        sub_names = sorted(sub_names)
        val_subs = sub_names[:int(len(sub_names)*cf.val_ratio)]
        test_subs = sub_names[int(len(sub_names)*cf.val_ratio):(int(len(sub_names)*cf.val_ratio)+int(len(sub_names)*cf.test_ratio))]
        train_subs = sub_names[(int(len(sub_names)*cf.val_ratio)+int(len(sub_names)*cf.test_ratio)):]

        tmpp = list(set(val_subs).intersection(test_subs,train_subs))
        assert len(tmpp) == 0
        
        from shock.dataset.HMC import HMCDataset
        train_dataset = HMCDataset(file_path=Path(cf.h5_data_path)/('HMC.hdf5'),sub_names=train_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        val_dataset = HMCDataset(file_path=Path(cf.h5_data_path)/('HMC.hdf5'),sub_names=val_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        test_dataset = HMCDataset(file_path=Path(cf.h5_data_path)/('HMC.hdf5'),sub_names=test_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        
        ch_list = train_dataset.get_ch_names()


    elif cf.dataset == 'shu_dataset':

        tmp = h5py.File(Path(cf.h5_data_path)/('shu_dataset.hdf5'),'r')
        sub_names_sessions = list(tmp.keys())
        
        sub_names = list(set([f.split("_")[0] for f in sub_names_sessions]))
        sub_names = sorted(sub_names)
        random.seed(214)
        random.shuffle(sub_names)
        random.seed(args.seed)
        tmp_train_subs = sub_names[:int(len(sub_names)*cf.train_ratio)]
        tmp_test_subs = sub_names[int(len(sub_names)*cf.train_ratio):(int(len(sub_names)*cf.test_ratio)+int(len(sub_names)*cf.train_ratio))]
        tmp_val_subs = sub_names[(int(len(sub_names)*cf.test_ratio)+int(len(sub_names)*cf.train_ratio)):]
        
        train_subs = [f for f in sub_names_sessions if f.split("_")[0] in tmp_train_subs]
        test_subs = [f for f in sub_names_sessions if f.split("_")[0] in tmp_test_subs]
        val_subs = [f for f in sub_names_sessions if f.split("_")[0] in tmp_val_subs]


        tmpp = list(set(val_subs).intersection(test_subs,train_subs))
        assert len(tmpp) == 0
        
        from shock.dataset.shu_dataset import shuDataset
        train_dataset = shuDataset(file_path=Path(cf.h5_data_path)/('shu_dataset.hdf5'),sub_names=train_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        val_dataset = shuDataset(file_path=Path(cf.h5_data_path)/('shu_dataset.hdf5'),sub_names=val_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        test_dataset = shuDataset(file_path=Path(cf.h5_data_path)/('shu_dataset.hdf5'),sub_names=test_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        
        ch_list = train_dataset.get_ch_names()


    elif cf.dataset == 'MODMA':
        labels = {'02010002':1, '02010004':1, '02010005':1, '02010006':1, '02010008':1, '02010010':1, '02010011':1, '02010012':1, '02010013':1, '02010015':1, '02010016':1, '02010018':1, '02010019':1, 
                '02010021':1, '02010022':1, '02010023':1, '02010024':1, '02010025':1, '02010026':1, '02010028':1, '02010030':1, '02010033':1, '02010034':1, '02010036':1, '02020008':0, '02020010':0,
                '02020013':0, '02020014':0, '02020015':0, '02020016':0, '02020018':0, '02020019':0, '02020020':0, '02020021':0, '02020022':0, '02020023':0, '02020025':0, '02020026':0, '02020027':0, 
                '02020029':0, '02030002':0, '02030003':0, '02030004':0, '02030005':0, '02030006':0, '02030007':0, '02030009':0, '02030014':0, '02030017':0, '02030018':0, '02030019':0, '02030020':0,
                '02030021':0,}
        tmp = h5py.File(Path(cf.h5_data_path)/('MODMA.hdf5'),'r')
        sub_names = list(tmp.keys())
        sub_names = sorted(sub_names)
        random.seed(214)
        random.shuffle(sub_names)
        random.seed(args.seed)
        train_subs = sub_names[:int(len(sub_names)*cf.train_ratio)]
        test_subs = sub_names[int(len(sub_names)*cf.train_ratio):(int(len(sub_names)*cf.test_ratio)+int(len(sub_names)*cf.train_ratio))]
        val_subs = sub_names[(int(len(sub_names)*cf.test_ratio)+int(len(sub_names)*cf.train_ratio)):]

        tmpp = list(set(val_subs).intersection(test_subs,train_subs))
        assert len(tmpp) == 0
        
        train_label = [labels[i] for i in train_subs]
        test_label = [labels[i] for i in test_subs]
        val_label = [labels[i] for i in val_subs]
        if logger is not None:
            logger.info(f'train_subs: {train_subs}')
            logger.info(f'val_subs: {val_subs}')
            logger.info(f'test_subs: {test_subs}')
            logger.info(f'train 1/0: {Counter(train_label)}, val 1/0: {Counter(val_label)}, tests 1/0: {Counter(test_label)}')
        
        from shock.dataset.MODMA import MODMADataset
        train_dataset = MODMADataset(file_path=Path(cf.h5_data_path)/('MODMA.hdf5'),sub_names=train_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        val_dataset = MODMADataset(file_path=Path(cf.h5_data_path)/('MODMA.hdf5'),sub_names=val_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        test_dataset = MODMADataset(file_path=Path(cf.h5_data_path)/('MODMA.hdf5'),sub_names=test_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        
        ch_list = train_dataset.get_ch_names()
        ch_list = [i.upper() for i in ch_list]        

    elif cf.dataset == 'EH':
        tmp = h5py.File(Path(cf.h5_data_path)/('EH.hdf5'),'r')
        sub_names = list(tmp.keys())
        sub_names = sorted(sub_names,reverse=True)
        sub_names_0 = []
        sub_names_1 = []
        all_labels = {}
        for sub in sub_names:
            label = tmp[sub][sub+'_label']
            label = int(np.array(label)) 
            all_labels[sub]=label
            if label == 0:
                sub_names_0.append(sub)
            else:
                sub_names_1.append(sub)
        
        random.seed(214)
        random.shuffle(sub_names_1)
        random.shuffle(sub_names_0)
        
        train_subs = sub_names_1[:int(len(sub_names_1)*cf.train_ratio)]+sub_names_0[:int(len(sub_names_0)*cf.train_ratio)]
        test_subs = sub_names_1[int(len(sub_names_1)*cf.train_ratio):(int(len(sub_names_1)*cf.test_ratio)+int(len(sub_names_1)*cf.train_ratio))]+\
                    sub_names_0[int(len(sub_names_0)*cf.train_ratio):(int(len(sub_names_0)*cf.test_ratio)+int(len(sub_names_0)*cf.train_ratio))]
        val_subs = sub_names_1[(int(len(sub_names_1)*cf.test_ratio)+int(len(sub_names_1)*cf.train_ratio)):] +\
                    sub_names_0[(int(len(sub_names_0)*cf.test_ratio)+int(len(sub_names_0)*cf.train_ratio)):]
        random.shuffle(train_subs)
        random.shuffle(test_subs)
        random.shuffle(val_subs)
        random.seed(args.seed)
                    
        tmpp = list(set(val_subs).intersection(test_subs,train_subs))
        assert len(tmpp) == 0
        train_label = [all_labels[i] for i in train_subs]
        test_label = [all_labels[i] for i in test_subs]
        val_label = [all_labels[i] for i in val_subs]
        if logger is not None:
            logger.info(f'train_subs: {train_subs}')
            logger.info(f'val_subs: {val_subs}')
            logger.info(f'test_subs: {test_subs}')
            logger.info(f'train 1/0: {Counter(train_label)}, val 1/0: {Counter(val_label)}, tests 1/0: {Counter(test_label)}')
        
        from shock.dataset.EH import EHDataset
        train_dataset = EHDataset(file_path=Path(cf.h5_data_path)/('EH.hdf5'),sub_names=train_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        val_dataset = EHDataset(file_path=Path(cf.h5_data_path)/('EH.hdf5'),sub_names=val_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        test_dataset = EHDataset(file_path=Path(cf.h5_data_path)/('EH.hdf5'),sub_names=test_subs, window_size=cf.window_size, stride_size=cf.stride_size)
        
        ch_list = train_dataset.get_ch_names()
        ch_list = [i.upper() for i in ch_list]  



    return train_dataset, val_dataset, test_dataset, ch_list, test_ch_list