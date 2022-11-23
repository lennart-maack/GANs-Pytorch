import h5py
import numpy as np
from pathlib import Path
import torch
from torch.utils import data
from PIL import Image

class HDF5Dataset(data.Dataset):
    """Represents an abstract HDF5 dataset. The data is loaded immediately into RAM.
    For bigger datasets, consider implementing a cache as in https://towardsdatascience.com/hdf5-datasets-for-pytorch-631ff1d750f5
    
    Input params:
        file_path: Path to the folder containing the dataset (one HDF5 file).
        transform: PyTorch transform to apply to every data instance (default=None).
    """
    def __init__(self, file_path, data_cache_size=3, transform=None):
        super().__init__()
        self.data_info = []
        self.transform = transform
        self.file_path = file_path

        # Search for all h5 files
        p = Path(file_path)
        assert(p.is_dir())
        files = sorted(p.glob('*.h5'))
        if len(files) < 1:
            raise RuntimeError('No hdf5 datasets found')
        
        for h5dataset_fp in files:
            self._add_data_info(str(h5dataset_fp.resolve()))
            
    def __getitem__(self, index):
        # get data
        x = self._get_data("image", index)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)

        # get labels
        y = self._get_data("label", index)
        y = torch.from_numpy(y)
        return (x, y)

    def __len__(self):
        return len(self.data_info)
    
    def _add_data_info(self, file_path):
        with h5py.File(file_path) as h5_file:
            # Walk through all groups, extracting datasets
            for gname, group in h5_file.items():
                for sub_group_name, sub_group in group.items():
                    for dname, ds in sub_group.items():
                        if dname == "images":
                            for i in range(len(ds)):   
                                id = f'{file_path}/{gname}/{sub_group_name}/{dname}/{str(i)}'
                                label = group[sub_group_name]["labels"][i]
                                image = group[sub_group_name]["images"][i]
                                self.data_info.append({'id': id, 'label': label, 'gname' : gname, 'sub_group_name' : sub_group_name, 'image' : image})


    def _get_data(self, type, index):
        return self.data_info[index][type]