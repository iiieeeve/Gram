from shock.dataset.dataset import PreprocessDataset, ShockDataset
from pathlib import Path
import mne
from shock.dataset.h5 import h5Dataset
import numpy as np
import matplotlib.pyplot as plt
import difflib
import scipy.io as sio
import torch
"""
wakefulness, stages N1, N2, N3, and R
"""
ch_names = ["Fp1", "Fp2", "Fz", "F3", "F4", "F7", "F8", "FC1", "FC2", "FC5",
            "FC6", "Cz", "C3", "C4", "T3", "T4", "A1", "A2", "CP1", "CP2",
            "CP5", "CP6", "Pz", "P3", "P4", "T5", "T6", "PO3", "PO4", "Oz",
            "O1", "O2"]
event_id = {'left': 0,'right':1}


class shu_dataset(PreprocessDataset):
           
    def create_raw(self,data,freq=250.0):

        si,sj,sk=data.shape
        da=data.transpose(1,0,2)
        da=da.reshape(sj,si*sk)
        llen=data.shape[0]
        ch_types = ['eeg'] * len(ch_names)
        info = mne.create_info(ch_names, freq, ch_types)
        raw = mne.io.RawArray(da, info)  # create raw
        raw = self._eeg_resample(raw)

        return raw

    def mnebandResample(self,data,labels,lowfreq,highfreq):
        data = self.create_raw(data,labels)
        data.filter(lowfreq, highfreq, fir_design='firwin')
        event = data.info['events']
        train_epoches = mne.Epochs(data, event, event_id, 0, 4 - 0.004,
                                baseline=None, preload=True)
        train_data = train_epoches.get_data()
        return train_data
    
    def preprocess_mat(self, data_path: 'Path|str', drop_list: list = []):
        """
        处理edf格式的数据
        """
        da=sio.loadmat(data_path)
        data=da['data']
        labels=np.ravel(da['labels'])
        si,sj,sk=data.shape
        raw = self.create_raw(data)
        eeg = raw.get_data()
        eeg = eeg.reshape(32,si,-1)
        eeg = eeg.transpose(1,0,2)

        
        ch_list = [i.upper() for i in ch_names]
        labels = labels - 1


        return ch_list, eeg, labels



    def generate_h5(self, data_paths: Path, save_path: Path, save_name: str):
        """生成h5文件, 需要被重写

        :param Path data_paths: 数据路径
        :param Path save_path: h5文件保存路径
        :param str save_name: h5文件保存名
        """
        # 若已有h5文件存在，获取已处理好的被试名，不再重复处理
        # exist_subs = []
        # if (save_path / save_name).exists():
        #     h5_file = h5Dataset(save_path, save_name, mode='r')
        #     exist_subs = h5_file.get_group_names()
        #     h5_file.save()

        
        for data_path in data_paths.glob('*.mat'):
            if '_sleepscoring' in data_path.stem:
                continue
            sub_name = data_path.stem.split('_')[0]+'_'+ data_path.stem.split('_')[1]

    
            print(f'preprocessing {sub_name}')
            try:
                ch_list, eeg, label = self.preprocess_mat(data_path)
                chunks = (len(ch_list), self.resample_rate)
            except Exception as e:
                print(f'{sub_name} has something wrong with the data')
                print(e)
                break

            h5_file = h5Dataset(save_path, save_name)
            exist_subs = h5_file.get_group_names()
            if sub_name not in exist_subs:
                grp = h5_file.addGroup(grpName=sub_name)
            else:
                grp = h5_file.get_group(grpName=sub_name)
                
                
            exist_files = list(grp.keys())

            for idd,(one_eeg, one_label) in enumerate(zip(eeg,label)):
                file_name = sub_name+'_'+str(idd)
                if file_name in exist_files:
                    print(f'{file_name} has been preprocessed')
                    continue
            
                
                dset = h5_file.addDataset(grp, file_name, one_eeg, chunks=chunks)
                # 为dataset添加基础的预处理属性
                self.bandFilter = [0.5,40]
                h5_file.addBaseAttributes(dset, ch_list, self.bandFilter, self.notchFilter, self.resample_rate)
                
                label_dset = h5_file.addDataset(grp, file_name+'_label', one_label,chunks=None)
                h5_file.addAttributes(label_dset,'label_type',f'{event_id}')


            # 关闭h5文件
            h5_file.save()

class shuDataset(ShockDataset):
    def get_label(self, subject_id, file_id, item_start_idx):
        sub_name = self.subjects[subject_id]
        file_name = self.sub_files_update[sub_name][file_id]
        label_file_name = file_name+'_label' 
        #each clip is annotated as a positive sample if all the three experts annotated the presence of seizure
        label = self.file[sub_name][label_file_name]
        label = np.array(label)
        # ttmp = 1 if 1 in label else 0
        
        return label

if __name__ =='__main__':
    

    # data_paths = Path(f'/shu_dataset/19228725/mat_files/mat')
    # save_path = Path(f'/h5Data')
    # save_name = f'shu_dataset'
    # preDataset=shu_dataset(sample_rate=250, notchFilter=50.0)
    # preDataset.generate_h5(data_paths,save_path,save_name)

    tmp = shuDataset(file_path='/h5Data/shu_dataset.hdf5')
    tmp_loader = torch.utils.data.DataLoader(tmp, batch_size=16,drop_last=True, shuffle=True)
    data,label = next(iter(tmp_loader))
    print(data.shape)
    print(data[0])
    print(label)

# torch.Size([16, 32, 200])
# tensor([[ 1.1834,  1.6706,  1.1089,  ...,  1.8926,  1.3661,  0.6687],
#         [ 1.3287,  1.8140,  1.4701,  ...,  1.3843,  1.3216,  0.9528],
#         [ 0.3453,  0.6976,  0.8702,  ...,  1.3151,  1.1531,  0.7469],
#         ...,
#         [-0.1785, -1.7845, -0.2418,  ...,  0.5186, -1.0312, -0.3484],
#         [ 0.6416, -0.9568,  0.9066,  ...,  0.4655, -0.8495, -0.2234],
#         [-0.0879, -1.8246, -0.9377,  ...,  1.9877, -0.4066, -0.4495]],
#        dtype=torch.float64)
# tensor([0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1])
