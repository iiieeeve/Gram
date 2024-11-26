from .dataset import PreprocessDataset,ShockDataset
from pathlib import Path
import mne
from .h5 import h5Dataset
import numpy as np
import torch

class TUEV(PreprocessDataset):
        
    def rename_ch_list(self, raw: mne.io.Raw):
        ch_rename_dict = {}
        ref = raw.ch_names[0].split('-')[1]
        ref = ref.upper()
        for ch in raw.ch_names:
            new_name = ch[4:].split('-')[0]
            new_name = new_name.upper()
            ch_rename_dict.update({ch:new_name})

        if ch_rename_dict != None:
            raw.rename_channels(ch_rename_dict)
        return raw

    def preprocess_edf(self, data_path: 'Path|str', drop_list: list = []):

        raw = mne.io.read_raw_edf(data_path, preload=True)


        remain_chs = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 
                    'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
                    'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 
                    'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'] 
        drop_list = list(set(raw.ch_names)-set(remain_chs))
        
        if len(drop_list) != 0:
            for i in drop_list:
                if i in raw.ch_names:
                    raw.drop_channels([i])
        assert len(raw.ch_names)==19
        
        raw = self.rename_ch_list(raw)  
        raw = self._eeg_reference(raw)
        raw = self._eeg_resample(raw)
        raw = self._eeg_filter(raw)
        

        ch_list = raw.ch_names
        eeg = raw.get_data(units='uV')
        
        remain_chs_rank = {'FP1':1, 'FP2':2, 'F3':3, 'F4':4, 'C3':5, 'C4':6, 'P3':7, 'P4':8, 'O1':9, 'O2':10,'F7':11, 'F8':12, 
                           'T3':13, 'T4':14, 'T5':15, 'T6':16,'FZ':17, 'CZ':18, 'PZ':19}
        

        ch_id = []
        for tmp in ch_list:
            ch_id.append(remain_chs_rank[tmp])
        rank_ch_list = sorted(range(len(ch_id)),key=lambda x: ch_id[x])
        eeg=np.take(eeg,rank_ch_list,axis=0)
        
        times = raw.times
        ww,tt = raw[:]
        RecFile_path = data_path.parent/(data_path.stem+".rec")
        eventData = np.genfromtxt(RecFile_path, delimiter=",")
        sfreq = raw.info['sfreq']

        return eeg, times, eventData, remain_chs, sfreq


    def BuildEvents(self, signals, times, EventData,sfreq):
        [numEvents, z] = EventData.shape  # numEvents is equal to # of rows of the .rec file
        [numChan, numPoints] = signals.shape
        # for i in range(numChan):  # standardize each channel
        #     if np.std(signals[i, :]) > 0:
        #         signals[i, :] = (signals[i, :] - np.mean(signals[i, :])) / np.std(signals[i, :])
        features = np.zeros([numEvents, numChan, int(sfreq) * 5])
        offending_channel = np.zeros([numEvents, 1])  # channel that had the detected thing
        labels = np.zeros([numEvents, 1])
        offset = signals.shape[1]
        signals = np.concatenate([signals, signals, signals], axis=1)
        for i in range(numEvents):  # for each event
            chan = int(EventData[i, 0])  # chan is channel
            start = np.where((times) >= EventData[i, 1])[0][0]
            end = np.where((times) >= EventData[i, 2])[0][0]
            # print (offset + start - 2 * int(fs), offset + end + 2 * int(fs), signals.shape)
            features[i, :] = signals[
                :, offset + start - 2 * int(sfreq) : offset + end + 2 * int(sfreq)
            ]
            offending_channel[i, :] = int(chan)
            labels[i, :] = int(EventData[i, 3]-1)
        return [features, offending_channel, labels]


    def generate_h5(self, data_pathss: Path, save_path: Path, save_name: str):

        
        label_map = { 0: 'spsw',1: 'gped',2:'pled',3: 'eyem',4: 'artf',5: 'bckg'}

        
        for data_paths in data_pathss.iterdir():
            for data_path in data_paths.glob('*.edf'):
                file_name = data_path.stem
                sub_name = data_paths.stem        
                
                print(f'preprocessing {file_name}')
                try:
                    eeg, times, eventData, ch_list, sfreq = self.preprocess_edf(data_path)
                    [signals, offending_channels, labels] = self.BuildEvents(eeg, times,eventData,sfreq)
                    chunks = (len(ch_list), self.resample_rate)
                except Exception as e:
                    print(f'{file_name} has something wrong with the data')
                    print(e)
                    break

                h5_file = h5Dataset(save_path, save_name)
                exist_subs = h5_file.get_group_names()
                if sub_name not in exist_subs:
                    grp = h5_file.addGroup(grpName=sub_name)
                else:
                    grp = h5_file.get_group(grpName=sub_name)
                    
                    
                exist_files = list(grp.keys())

                for idx,(signal,offending_channel, label) in enumerate(zip(signals, offending_channels, labels)):
                    seg_file_name = file_name+f'_{idx}'
                    if seg_file_name in exist_files:
                        print(f'{file_name} has been preprocessed')
                        continue
                    
                    dset = h5_file.addDataset(grp, seg_file_name, signal, chunks=chunks)

                    h5_file.addBaseAttributes(dset, ch_list, self.bandFilter, self.notchFilter, self.resample_rate)
                    
                    label_dset = h5_file.addDataset(grp, seg_file_name+'_label', label, chunks=None)
                    h5_file.addAttributes(label_dset,'label_type',f'{label_map}')



                h5_file.save()


class TUEVDataset(ShockDataset):
    def get_label(self, subject_id, file_id, item_start_idx):
        sub_name = self.subjects[subject_id]
        file_name = self.sub_files_update[sub_name][file_id]
        label_file_name = file_name+'_label'
        label = self.file[sub_name][label_file_name]
        label = np.array(label)
        return label[0]

if __name__ =='__main__':
    
    for flag1 in ['eval','train']:
        data_paths = Path(f'your path/TUEV/edf/{flag1}')
        save_path = Path(f'your path')
        save_name = f'TUEV_{flag1}'
        preDataset=TUEV(sample_rate=300, notchFilter=60.0)
        preDataset.generate_h5(data_paths,save_path,save_name)
    
    tmp = TUEVDataset(file_path='your path')
    tmp_loader = torch.utils.data.DataLoader(tmp, batch_size=16,drop_last=True, shuffle=True)
    data,label = next(iter(tmp_loader))
    print(data.shape)
    print(label)
