import h5py
import bisect
from pathlib import Path
from typing import List

# import torch
from torch.utils.data import Dataset


list_path = List[Path]

class SingleShockDataset(Dataset):

    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):

        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []

  
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx)
            subject_len = self.__file[subject]['eeg'].shape[1]

            total_sample_num = (subject_len-self.__window_size) // self.__stride_size + 1

            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size 
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        return self.__file[self.__subjects[subject_idx]]['eeg'][:, item_start_idx:item_start_idx+self.__window_size]
    
    def free(self) -> None: 

        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']


class ShockDataset(Dataset):

    def __init__(self, file_paths: list_path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):

        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets = []
        self.__length = None
        self.__feature_size = None

        self.__dataset_idxes = []


        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage, self.__end_percentage) for file_path in self.__file_paths]
        

        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self):
        return self.__feature_size

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]
    
    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()
    
    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()


class DownstreamDataset(Dataset):

    def __init__(self, file_path: Path, label_map=None, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1, trial_start_percentage: float=0, trial_end_percentage: float=1, subject_start_percentage: float=0, subject_end_percentage: float=1):

        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage
        self.__trial_start_percentage = trial_start_percentage
        self.__trial_end_percentage = trial_end_percentage
        self.__subject_start_percentage = subject_start_percentage
        self.__subject_end_percentage = subject_end_percentage

        self.__file = None
        self.__length = None
        self.__feature_size = None

        self.__subjects = []
        self.__global_idxes = [] 
        self.__local_idxess = [] 
        self.__trial_start_idxess = [] 
        self.__genders = []
        self.__labelss = []
        self.label_map=label_map

        self.__rsFreq = None
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        subject_start_id = int(len(self.__subjects) * self.__subject_start_percentage) 
        subject_end_id = int(len(self.__subjects) * self.__subject_end_percentage - 1) 
        for subject_id, subject in enumerate(self.__subjects):
            self.__global_idxes.append(global_idx)
            self.__genders.append(self.__file[subject].attrs['gender'])
            if self.label_map is not None:
                tmp = [self.label_map[i] for i in self.__file[subject].attrs['label']]
            else:
                tmp = self.__file[subject].attrs['label']
            self.__labelss.append(tmp)
            self.__rsFreq = self.__file[subject]['eeg'].attrs['rsFreq']

            local_idxes = [] 
            trial_start_idxes = [] 
            trial_starts = self.__file[subject].attrs['trialStart']
            trial_ends = self.__file[subject].attrs['trialEnd']
            local_idx = 0
            if subject_id >= subject_start_id and subject_id <= subject_end_id:
                trial_start_id = int(len(trial_starts) * self.__trial_start_percentage)  
                trial_end_id = int(len(trial_starts) * self.__trial_end_percentage - 1)  
                for trial_id, (trial_start, trial_end) in enumerate(zip(trial_starts, trial_ends)):
                    local_idxes.append(local_idx)

                    if trial_id >= trial_start_id and trial_id <= trial_end_id:
                        trial_len = (trial_end - trial_start + 1) * self.__rsFreq
                        trial_sample_num = (trial_len-self.__window_size) // self.__stride_size + 1
                        start_idx = int(trial_sample_num * self.__start_percentage) * self.__stride_size + trial_start * self.__rsFreq
                        end_idx = int(trial_sample_num * self.__end_percentage - 1) * self.__stride_size + trial_start * self.__rsFreq

                        trial_start_idxes.append(start_idx)
                        local_idx += (end_idx - start_idx) // self.__stride_size + 1
                    else:
                        trial_start_idxes.append(0)

            self.__local_idxess.append(local_idxes)
            self.__trial_start_idxess.append(trial_start_idxes)

            global_idx += local_idx

        self.__length = global_idx

        self.__feature_size = [i for i in self.__file[self.__subjects[0]]['eeg'].shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self):
        return self.__feature_size
    
    @property
    def rsfreq(self):
        return self.__rsFreq

    def __len__(self):
        return self.__length

    def __getitem__(self, idx: int):

        subject_id = bisect.bisect(self.__global_idxes, idx) - 1
        trial_id = bisect.bisect(self.__local_idxess[subject_id], idx-self.__global_idxes[subject_id]) - 1
        item_start_idx = (idx - self.__global_idxes[subject_id] - self.__local_idxess[subject_id][trial_id]) * self.__stride_size + self.__trial_start_idxess[subject_id][trial_id]


        labels = self.__labelss[subject_id][trial_id]
        data = self.__file[self.__subjects[subject_id]]['eeg'][:, item_start_idx:item_start_idx+self.__window_size]

        return data, labels
    
    def free(self) -> None: 

        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        return self.__file[self.__subjects[0]]['eeg'].attrs['chOrder']
    
    
if __name__ == '__main__':
    label_map = {'H':0,'F':1,'N':2,'S':3,'D':4}
    train_dataset = DownstreamDataset(Path('your path')/('seed-5.hdf5'),label_map)
