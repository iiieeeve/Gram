import h5py
import numpy as np
from pathlib import Path

class h5Dataset:
    def __init__(self, path:Path, name:str,mode:str='a') -> None:
        self.__name = name
        if mode !='a' and mode !='r':
            raise Exception(f'can not set mode to {mode}, only "a" or "r"')
        self.__f = h5py.File(path / f'{name}.hdf5', mode)
        
    def get_group_names(self):
        return list(self.__f.keys())

    def get_dataset_names_from_group(self,grpName:h5py.Group):
        return list(self.__f[grpName].keys())

    def get_group(self,grpName:h5py.Group):
        return self.__f[grpName]

    def get_dataset_from_group(self,grpName:h5py.Group,dsName:h5py.Dataset):
        return self.__f[grpName][dsName]
    
    def addGroup(self, grpName:str):
        return self.__f.create_group(grpName)
    
    def addDataset(self, grp:h5py.Group, dsName:str, arr:np.array, chunks:tuple):
        if chunks is not None:
            return grp.create_dataset(dsName, data=arr, chunks=chunks)
        else:
            return grp.create_dataset(dsName, data=arr)

    def addAttributes(self, src:'h5py.Dataset|h5py.Group', attrName:str, attrValue):
        src.attrs[f'{attrName}'] = attrValue

    def addBaseAttributes(self, src:'h5py.Dataset',ch_list, bandFilter=[0,75],notchFilter=50.0,resample_rate=200):
        src.attrs[ 'lFreq'] = bandFilter[0]
        src.attrs[ 'hFreq'] = bandFilter[1]
        src.attrs[ 'nFreq'] = notchFilter
        src.attrs[ 'rsFreq'] = resample_rate
        src.attrs[ 'chOrder'] = ch_list

    def save(self):
        self.__f.close()
    
    @property
    def name(self):
        return self.__name

