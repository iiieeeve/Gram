from shock.dataset.dataset import PreprocessDataset,ShockDataset
from pathlib import Path
import mne
from shock.dataset.h5 import h5Dataset
import numpy as np
import matplotlib.pyplot as plt
import difflib
from copy import deepcopy
import torch
"""
wakefulness, stages N1, N2, N3, and R
"""
annotation_desc_2_event_id = {
    "Sleep stage W": 1,
    "Sleep stage N1": 2,
    "Sleep stage N2": 3,
    "Sleep stage N3": 4,
    "Sleep stage R": 5,
}



class HMC(PreprocessDataset):
        
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

    def get_label(self,data_path:str):
        """
        """
        label = None
        return label
            
            
        

    def preprocess_edf(self, data_path: 'Path|str', drop_list: list = []):

        raw = mne.io.read_raw_edf(data_path, preload=True)
        remain_chs = ['EEG F4-M1', 'EEG C4-M1', 'EEG O2-M1', 'EEG C3-M2']  
        drop_list = list(set(raw.ch_names)-set(remain_chs))
        
        if len(drop_list) != 0:
            for i in drop_list:
                if i in raw.ch_names:
                    raw.drop_channels([i])
        assert len(raw.ch_names)==4
        
        raw = self.rename_ch_list(raw)  
        raw = self._eeg_resample(raw)
        raw = self._eeg_filter(raw)
        new_ch_list = ['F4', 'C4', 'O2', 'C3']
        raw.reorder_channels(new_ch_list)


        label_path = data_path.parent/(data_path.stem+'_sleepscoring.edf')
        annot_train = mne.read_annotations(label_path)


        tmp = difflib.get_close_matches('Lights', annot_train.description,cutoff=0.2)
        assert len(tmp)==2
        off_string = tmp[0] if 'off' in tmp[0] else tmp[1]
        on_string = tmp[0] if 'on' in tmp[0] else tmp[1]
        print('==='*50)
        print(f'off:{off_string}')
        print(f'on:{on_string}')
        assert len(annot_train.onset[annot_train.description==off_string]) == 1
        assert len(annot_train.onset[annot_train.description==on_string]) == 1
        annot_train.crop(annot_train.onset[annot_train.description==off_string][0], annot_train.onset[annot_train.description==on_string][0])

        raw.set_annotations(annot_train, emit_warning=False)
        anno_type = set(list(annot_train.description))
        e_id = deepcopy(annotation_desc_2_event_id)
        ks = set(list(e_id.keys()))
        del_ks = ks - anno_type
        for k in del_ks:
            del e_id[k]
        print(e_id)

        events_train, _ = mne.events_from_annotations(raw, event_id=e_id,chunk_duration=30.0)
        tmax = 30.0 - 1.0 / raw.info["sfreq"]  # tmax in included
        epochs_train = mne.Epochs(
                        raw=raw,
                        events=events_train,
                        event_id=e_id,
                        baseline=None,
                        tmin=0.0,
                        tmax=tmax,

                    )

        eeg = epochs_train.get_data(units='uV')
        label = epochs_train.events[:,-1]
        ch_list = epochs_train.ch_names
        

        return ch_list, eeg, label, e_id



    def generate_h5(self, data_paths: Path, save_path: Path, save_name: str):

        
        for data_path in data_paths.glob('*.edf'):
            if '_sleepscoring' in data_path.stem:
                continue
            sub_name = data_path.stem   


            print(f'preprocessing {sub_name}')
            try:
                ch_list, eeg, label, e_id = self.preprocess_edf(data_path)
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

                h5_file.addBaseAttributes(dset, ch_list, self.bandFilter, self.notchFilter, self.resample_rate)
                
                label_dset = h5_file.addDataset(grp, file_name+'_label', one_label,chunks=None)
                h5_file.addAttributes(label_dset,'label_type',f'{e_id}')



            h5_file.save()


class HMCDataset(ShockDataset):
    def get_label(self, subject_id, file_id, item_start_idx):
        sub_name = self.subjects[subject_id]
        file_name = self.sub_files_update[sub_name][file_id]
        label_file_name = file_name+'_label' 
        #each clip is annotated as a positive sample if all the three experts annotated the presence of seizure
        label = self.file[sub_name][label_file_name]
        label = np.array(label)
        # ttmp = 1 if 1 in label else 0
        #"{'Sleep stage W': 1, 'Sleep stage N1': 2, 'Sleep stage N2': 3, 'Sleep stage N3': 4, 'Sleep stage R': 5}"
        
        return label-1




if __name__ =='__main__':


    data_paths = Path(f'your path')
    save_path = Path(f'your path')
    save_name = f'HMC'
    preDataset=HMC(sample_rate=256, notchFilter=50.0)
    preDataset.generate_h5(data_paths,save_path,save_name)

    tmp = HMCDataset(file_path='your path')
    tmp_loader = torch.utils.data.DataLoader(tmp, batch_size=16,drop_last=True, shuffle=True)
    data,label = next(iter(tmp_loader))
    print(data.shape)
    print(data[0,0])
    print(label)



