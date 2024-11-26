import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from shock.dataset.h5 import h5Dataset
from torch.utils.data import Dataset
import h5py
import bisect


class PreprocessDataset:
    def __init__(self, resample_rate=200, bandFilter=[0, 75], notchFilter=50.0, **args):
        self.resample_rate = resample_rate
        self.bandFilter = bandFilter
        self.notchFilter = notchFilter

    def _eeg_resample(self, raw: mne.io.Raw):

        if self.resample_rate != raw.info['sfreq']:
            raw = raw.resample(self.resample_rate)

        return raw

    def _eeg_filter(self, raw: mne.io.Raw) -> mne.io.Raw:


        if self.bandFilter != None:
            raw = raw.filter(l_freq=self.bandFilter[0], h_freq=self.bandFilter[1])

        if self.notchFilter != None:
            raw = raw.notch_filter(self.notchFilter)

        return raw

    def _eeg_reference(self, raw: mne.io.Raw, ref_channels='average', original_ref_channel=None):

        if original_ref_channel != None:
            if original_ref_channel not in raw.ch_names:
                raise Exception(f"original_ref_channel not in ch_names, {original_ref_channel}")
            raw.drop_channels(ch_names=original_ref_channel)
        if ref_channels != None:
            if ref_channels == 'average' or ref_channels == 'REST':
                raw.set_eeg_reference(ref_channels, ch_type='eeg')
            elif isinstance(ref_channels, list):
                raw.set_eeg_reference(ref_channels, ch_type='eeg')
                raw.drop_channels(ch_names=ref_channels)
            else:
                raise Exception(f"Invalid ref_channels , {ref_channels} ")
        return raw

    def rename_ch_list(self, raw: mne.io.Raw):
 
        ch_rename_dict = None
        if ch_rename_dict != None:
            raw.rename_channels(ch_rename_dict)
        return raw

    def draw_psd_topo(self, raw: mne.io.Raw, montage_path: 'Path|str'):
        montage = mne.channels.read_custom_montage(montage_path)
        raw.set_montage(montage)
        raw.plot()
        raw.plot_psd_topomap(show=True)
        plt.show()

    def preprocess_array(self, data: np.array, ch_list: list, sample_rate, ref_channels='average'):

        self.ch_list = ch_list
        ch_types = ['eeg'] * len(self.ch_list)
        info = mne.create_info(self.ch_list, self.sample_rate, ch_types)
        raw = mne.io.RawArray(data, info)

        raw = self._eeg_reference(raw, ref_channels)
        raw = self._eeg_resample(raw)
        raw = self._eeg_filter(raw)

        eeg = raw.get_data(units='uV')
        return ch_list, eeg

    def preprocess_cnt(self, data_path: 'Path|str', drop_list: list = [], ref_channels='average'):
   

        raw = mne.io.read_raw_cnt(data_path, preload=True)
        raw = self.rename_ch_list(raw)  # cnt自带的channel名可能不规范，需要重新命名

        if len(drop_list) != 0:
            for i in drop_list:
                if i in raw.ch_names:
                    raw.drop_channels([i])

        raw = self._eeg_reference(raw, ref_channels)
        raw = self._eeg_resample(raw)
        raw = self._eeg_filter(raw)
        

        ch_list = raw.ch_names
        eeg = raw.get_data(units='uV')
        return ch_list, eeg

    def preprocess_edf(self, data_path: 'Path|str', drop_list: list = [], ref_channels='average'):

        raw = mne.io.read_raw_edf(data_path, preload=True)
        raw = self.rename_ch_list(raw)  # edf自带的channel名可能不规范，需要重新命名

        if len(drop_list) != 0:
            for i in drop_list:
                if i in raw.ch_names:
                    raw.drop_channels([i])

        raw = self._eeg_reference(raw, ref_channels)
        raw = self._eeg_resample(raw)
        raw = self._eeg_filter(raw)
        

        ch_list = raw.ch_names
        eeg = raw.get_data(units='uV')
        return ch_list, eeg

    def generate_h5(self, data_paths: Path, save_path: Path, save_name: str):

        exist_files = []
        if (save_path / (save_name+'.hdf5')).exists():
            h5_file = h5Dataset(save_path, save_name, mode='r')
            exist_subs = h5_file.get_group_names()
            h5_file.save()

        for data_path in data_paths.iterdir():
            h5_file = h5Dataset(save_path, save_name)
            sub_name = data_path.stem
            print(f'preprocessing {sub_name}')

            if data_path.stem in exist_files:
                print(f'{data_path.stem} has been preprocessed')
                h5_file.save()
                continue


            ch_list, eeg = self.preprocess_edf(data_path)
            chunks = (len(ch_list), self.resample_rate)

            grp = h5_file.addGroup(grpName=sub_name)
            dset = h5_file.addDataset(grp, 'eeg', eeg, chunks)

            h5_file.addBaseAttributes(dset, ch_list, self.bandFilter, self.notchFilter, self.resample_rate)


            h5_file.save()


#跨被试
class ShockDataset(Dataset):

    def __init__(self, file_path: Path, sub_files={},sub_names=[], window_size: int=200, stride_size: int=100, label_delay=0, remove_value=0,sample_rate=200):

        self.file_path = file_path
        self.window_size = window_size
        self.stride_size = stride_size
        self.label_delay = label_delay   
        self.remove_value = remove_value
        self.sample_rate = sample_rate
        self.sub_names = sub_names
        self.sub_files = sub_files  
        self.sub_files_update = {} 

        self.file = None
        self.length = None
        self.feature_size = None

        self.subjects = []
        self.global_idxes = []
        self.local_idxess = [] 


        self.rsFreq = None
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        print('init_dataset')
        self.file = h5py.File(str(self.file_path), 'r')
        if len(self.sub_names) == 0:
            self.subjects = [i for i in self.file]
        else:
            self.subjects = self.sub_names

        global_idx = 0
        for subject_id, subject in enumerate(self.subjects):
            self.global_idxes.append(global_idx)

            local_idxes = [] 


            local_idx = 0

            if len(self.sub_files) == 0:
                tmp_files = []
                for ttmp in self.file[subject].keys():
                    if 'label' not in ttmp:
                        tmp_files.append(ttmp)
            else:
                tmp_files = self.sub_files[subject]
            self.sub_files_update[subject] = tmp_files  
            for file in tmp_files:
                local_idxes.append(local_idx)

                parad_len = self.file[subject][file].shape[1]
                if self.remove_value > 0:
                    parad_sample_num = (parad_len-self.remove_value-self.window_size) // self.stride_size + 1
                else:
                    parad_sample_num = (parad_len-self.window_size) // self.stride_size + 1


                local_idx += parad_sample_num


            self.local_idxess.append(local_idxes)


            global_idx += local_idx

        self.length = global_idx


        print('dataset finish')


    def get_label(self, subject_id, file_id, item_start_idx):

        label = -1.0
        return label
    
    @property
    def rsfreq(self):
        return self.rsFreq

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):

        subject_id = bisect.bisect(self.global_idxes, idx) - 1
        file_id = bisect.bisect(self.local_idxess[subject_id], idx-self.global_idxes[subject_id]) - 1
        sub_name = self.subjects[subject_id]
        file_name = self.sub_files_update[sub_name][file_id]
        item_start_idx = (idx - self.global_idxes[subject_id] - self.local_idxess[subject_id][file_id]) * self.stride_size 

        label = self.get_label(subject_id, file_id, item_start_idx)

        return self.file[sub_name][file_name][:, item_start_idx:item_start_idx+self.window_size], label
    
    def free(self) -> None: 

        if self.file:
            self.file.close()
            self.file = None
            
    def get_ch_names(self):
        tmp_file_names = list(self.file[self.subjects[0]].keys())
        tmp_file_names = [i for i in tmp_file_names if 'label' not in i]
        return self.file[self.subjects[0]][tmp_file_names[0]].attrs['chOrder']

# 被试依赖
class SubDepShockDataset(Dataset):

    def __init__(self, file_path: Path, sub_files={},sub_names=[], start_percentage: float=0, end_percentage: float=1, window_size: int=200, stride_size: int=100, label_delay=0, remove_value=0, sample_rate=200):

        self.file_path = file_path
        self.window_size = window_size
        self.stride_size = stride_size 
        self.label_delay = label_delay   
        self.remove_value = remove_value
        self.sample_rate = sample_rate
        self.start_percentage = start_percentage
        self.end_percentage = end_percentage
        self.sub_names = sub_names
        self.sub_files = sub_files 
        self.sub_files_update = {} 
       

        self.file = None
        self.length = None
        self.feature_size = None

        self.subjects = []
        self.global_idxes = []
        self.local_idxess = [] 
        self.file_start_idxess = []  


        self.rsFreq = None
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        print('init_dataset')
        self.file = h5py.File(str(self.file_path), 'r')
        if len(self.sub_names) == 0:
            self.subjects = [i for i in self.file]
        else:
            self.subjects = self.sub_names

        global_idx = 0
        for subject_id, subject in enumerate(self.subjects):
            self.global_idxes.append(global_idx)

            local_idxes = []
            file_start_idxes = [] 


            local_idx = 0

            if len(self.sub_files) == 0:
                tmp_files = []
                for ttmp in self.file[subject].keys():
                    if 'label' not in ttmp:
                        tmp_files.append(ttmp)
            else:
                tmp_files = self.sub_files[subject]
            self.sub_files_update[subject] = tmp_files  
            for file in tmp_files:
                local_idxes.append(local_idx)

                parad_len = self.file[subject][file].shape[1]
                parad_sample_num = (parad_len-self.window_size) // self.stride_size + 1

                start_idx = int(parad_sample_num * self.start_percentage)  * self.stride_size
                if self.remove_value > 0:
                    tmp =  (parad_len-self.remove_value-self.window_size) // self.stride_size + 1
                    end_idx_1 = int(parad_sample_num * self.end_percentage - 1)  * self.stride_size
                    end_idx_2 = int(tmp - 1)  * self.stride_size
                    if end_idx_1 < end_idx_2:
                        end_idx = end_idx_1
                    else:
                        end_idx = end_idx_2
                else:
                    end_idx = int(parad_sample_num * self.end_percentage - 1)  * self.stride_size

                file_start_idxes.append(start_idx)
                local_idx += (end_idx - start_idx) // self.stride_size + 1
                

            self.local_idxess.append(local_idxes)
            self.file_start_idxess.append(file_start_idxes)


            global_idx += local_idx

        self.length = global_idx


        print('dataset finish')


    def get_label(self, subject_id, file_id, item_start_idx):

        label = -1.0
        return label
    
    @property
    def rsfreq(self):
        return self.rsFreq

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):

        subject_id = bisect.bisect(self.global_idxes, idx) - 1
        file_id = bisect.bisect(self.local_idxess[subject_id], idx-self.global_idxes[subject_id]) - 1
        sub_name = self.subjects[subject_id]
        file_name = self.sub_files_update[sub_name][file_id]
        item_start_idx = (idx - self.global_idxes[subject_id] - self.local_idxess[subject_id][file_id]) * self.stride_size + self.file_start_idxess[subject_id][file_id]

        label = self.get_label(subject_id, file_id, item_start_idx)

        return self.file[sub_name][file_name][:, item_start_idx:item_start_idx+self.window_size], label
    
    def free(self) -> None: 
   
        if self.file:
            self.file.close()
            self.file = None
            
    def get_ch_names(self):
        tmp_file_names = list(self.file[self.subjects[0]].keys())
        return self.file[self.subjects[0]][tmp_file_names[0]].attrs['chOrder']
if __name__ == '__main__':
    data_paths = Path('')
    save_path = Path('')
    save_name = ''
    preDataset = PreprocessDataset(sample_rate=300)
    preDataset.generate_h5(data_paths, save_path, save_name)