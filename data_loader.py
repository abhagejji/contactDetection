import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader,Dataset
import os

class MyDataLoading_color(Dataset):         #3-channels
    # for fine grain control over how each img is preproceed
    def __init__(self, x, mount_point, color_channel,shadow_channel,batch_size = 16, transforms=None, img_col = "path_to_img", label_col = "label"):
        # self.data_paths = x
        # self.label = y
        self.df = x
        self.transform = transforms
        self.transforms = transforms
        self.batch_size = 16
        self.color = color_channel
        self.shadow =shadow_channel
        self.mount_point = mount_point
        self.img_col = img_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        # Load and preprocess individual
        # image_path = self.data_paths[idx]
        image_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_col])
        label = self.df.iloc[idx][self.label_col]
        try:
            with h5py.File(image_path,"r") as f:
                colorTensor = torch.tensor(np.array(f[self.color]), dtype= torch.float32)
                colorTensor = torch.permute(colorTensor, [2,0,1])
            x = colorTensor
            y = torch.tensor(label, dtype = torch.long)
        except:
            print("error in reading train file : ",image_path, self.color, self.shadow)
            x = torch.tensor(np.zeros([3,512,512]), dtype = torch.float32)
            y = torch.tensor(np.random.randint(0,2), dtype = torch.long)
        return x, y
    
class MyDataLoading_four_channel(Dataset):
    # for fine grain control over how each img is preproceed
    def __init__(self, x,mount_point,color_channel,shadow_channel,batch_size = 16, transforms=None, img_col = "path_to_img", label_col = "label"):
        self.df = x
        # self.label = y
        self.transform = transforms
        self.transforms = transforms
        self.batch_size = 16
        self.color = color_channel
        self.shadow =shadow_channel
        self.mount_point = mount_point
        self.img_col = img_col
        self.label_col = label_col


    def __len__(self):
        return len(self.df.index)

    def __getitem__(self, idx):
        # Load and preprocess individual
        # image_path = self.data_paths[idx]
        image_path = os.path.join(self.mount_point, self.df.iloc[idx][self.img_col])
        label = self.df.iloc[idx][self.label_col]
        try:
            with h5py.File(image_path,"r") as f:
                colorTensor = torch.tensor(np.array(f[self.color]), dtype= torch.float32)
                colorTensor = torch.permute(colorTensor, [2,0,1])
                shadowTensor = torch.tensor(np.array(f[self.shadow]), dtype= torch.float32).unsqueeze(-1)
                shadowTensor = torch.permute(shadowTensor, [2,0,1])
            x = torch.cat((colorTensor,shadowTensor), dim = 0)#TODO colorTensor #
            # x = colorTensor
            y = torch.tensor(label, dtype = torch.long)
        except:
            print("error in reading train file : ",image_path, self.color, self.shadow)
            x = torch.tensor(np.zeros([4,512,512]), dtype = torch.float32)
            y = torch.tensor(np.random.randint(0,2), dtype = torch.long)
        return x, y
