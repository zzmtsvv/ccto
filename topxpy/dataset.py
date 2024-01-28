import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import imageio.v2 as io
import numpy as np
import topxpy as top
import copy
from torch.utils.data import Dataset


Q_KEYS = ('Q_00', 'Q_01', 'Q_02', 'Q_11', 'Q_12', 'Q_22', 'is_gray', 'volfrac', 'is_broken')


def save_datapoint(img, data, i, dataset_folder, forced=False):
    pics_folder = os.path.join(dataset_folder, 'pics')
    Path(pics_folder).mkdir(parents=True, exist_ok=True)
    data_folder = os.path.join(dataset_folder, 'data')
    Path(data_folder).mkdir(parents=True, exist_ok=True)

    fname = str(i)
    full_path_pic = pics_folder +'/'+ fname + '.png'
    if not forced:
        while os.path.exists(full_path_pic):
            i += 1
            fname = str(i)
            full_path_pic = pics_folder +'/'+ fname + '.png'
    else:
        fname = str(i)
        full_path_pic = pics_folder +'/'+ fname + '.png'
    # finally saving pic
    
    plt.imsave(full_path_pic, img, cmap='gray')
    # and corresponding data
    df =  pd.DataFrame(data, index=[0])
    full_path_csv = data_folder+ '/'+ fname + '.csv'
    df.to_csv(full_path_csv)
    return i


def load_datapoint(filename, dataset_folder):
    pics_folder = os.path.join(dataset_folder, 'pics')
    data_folder = os.path.join(dataset_folder, 'data')

    filenumber = int(filename.split('.')[0])
    file_path = os.path.join(pics_folder, filename)
    img_loaded = np.asarray( io.imread(file_path)[:,:,0] / 255.0 )
    fname = str(filenumber)
    full_path_pic = pics_folder +'/'+ fname + '.png'
    full_path_csv = data_folder+ '/'+ fname + '.csv'

    df = pd.read_csv(full_path_csv)
    data = df.to_dict(orient='records')[0]
    return img_loaded, data, full_path_pic, full_path_csv



class StructureTransform:
    def img_transform(self, img):
        pass
    
    def data_transform(self, data):
        pass

    def condition(self):
        pass

    def __call__(self, sample: dict):
        if self.condition():
            sample = copy.deepcopy(sample) # otherwise the transform changes the original data sample (outside of the function)
            img, data = sample['image'], sample['data']
            
            img = self.img_transform(img)
            data = self.data_transform(data)

            sample['image'] = img
            sample['data'].update(data)
        return sample


class RandomTranspose(StructureTransform):
    def __init__(self, p=0.5):        
        self.p = p
    
    def condition(self):
        return np.random.rand() < self.p
    
    def img_transform(self, img):
        return img.T
    
    def data_transform(self, data):
        Q = top.crit.dict_to_Q(data)
        Q[[0,1], :] = Q[[1,0], :]
        Q[:, [0,1]] = Q[:, [1,0]]
        data_new = top.crit.Q_to_dict(Q)
        return data_new


class RandomFlipUD(StructureTransform):
    def __init__(self, p=0.5):
        self.p = p
    
    def condition(self):
        return np.random.rand() <= self.p
    
    def img_transform(self, img):
        return np.flipud(img)
    
    def data_transform(self, data):
        data['Q_02'] = -data['Q_02']
        data['Q_12'] = -data['Q_12']
        return data


class RandomFlipLR(StructureTransform):
    def __init__(self, p=0.5):
        self.p = p
    
    def condition(self):
        return np.random.rand() <= self.p
    
    def img_transform(self, img):
        return np.fliplr(img)
    
    def data_transform(self, data):
        data['Q_02'] = -data['Q_02'] # FIX_ME
        data['Q_12'] = -data['Q_12']
        return data


class RandomRoll(StructureTransform):
    def condition(self):
        return True
    
    def img_transform(self, img):
        nx, ny = img.shape
        shift_x = np.random.randint(-nx, nx + 1)
        shift_y = np.random.randint(-ny, ny + 1)
        img = np.roll(img, shift_x, axis=0)
        img = np.roll(img, shift_y, axis=1)
        return img
    
    def data_transform(self, data):
        return data


class VectorizedStructuresDataset(Dataset):
    def __init__(self, dataset_folder, transform=None):
        self.dataset_folder = dataset_folder
        self.pics_folder = os.path.join(dataset_folder, 'pics')
        self.data_folder = os.path.join(dataset_folder, 'data')
        self.img_files = os.listdir(self.pics_folder)
        self.transform = transform

    def _load_datapoint(self, idx):
        img_filename = self.img_files[idx]
        img_loaded, data, full_path_pic, full_path_csv = top.dataset.load_datapoint(img_filename, self.dataset_folder)
        return img_loaded, data

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_loaded, data = self._load_datapoint(idx)

        sample = {'image': img_loaded, 'data': data}

        if self.transform:
            sample = self.transform(sample)       
        
        vectorized_data = top.crit.dict_to_vec(sample['data'])
        
        return sample['image'], vectorized_data


class VectorizedStructuresDatasetOnefile(VectorizedStructuresDataset):
    def __init__(self, dataset_folder, transform=None):
        super().__init__(dataset_folder, transform)
        self.data = pd.read_csv(os.path.join(self.data_folder, 'dataset.csv'))

    def _load_datapoint(self, idx):
            datapoint = self.data.loc[idx].to_dict()
            img_name = datapoint['img_name']
            img_file_path = os.path.join(self.pics_folder, img_name)
            img_loaded = np.asarray( io.imread(img_file_path)[:,:,0] / 255.0 )          
            return img_loaded, datapoint


def accumulate_csvs(data_folder):        
    data_files = sorted(os.listdir(data_folder))
    all_data = []

    for dfname in data_files:
        if not dfname == 'dataset.csv':
            data = pd.read_csv(os.path.join(data_folder, dfname))
            data['img_name'] = os.path.splitext(dfname)[0] + '.png'
            all_data.append(data)

    all_data = pd.concat(all_data)
    del all_data['Unnamed: 0']

    all_data.index = np.arange(len(all_data))
    all_data.to_csv( os.path.join(data_folder, 'dataset.csv'))