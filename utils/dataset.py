import numpy as np
import torch
import torch.utils.data as data
import pandas as pd
import utils.tools as tools
import torch.nn.functional as F
import os

class UCFDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict, normal: bool = False,pseudo_path1=None,pseudo_path2=None):
        self.df = pd.read_csv(file_path) # 일단 학습 데이터의 리스트를 들고 옴옴
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        self.normal = normal
        # raw_data1 = np.load('/SSD2/DY/VadCLIP/exp_ucf/pseudo4/pseudo1_ab_ucf.npz', allow_pickle=True)["data"]
        # raw_data2 = np.load('/SSD2/DY/VadCLIP/exp_ucf/pseudo4/pseudo1_nor_ucf.npz', allow_pickle=True)["data"]
        # raw_data3 = np.load('/SSD2/DY/VadCLIP/exp_ucf/pseudo4/pseudo2_ab_ucf.npz', allow_pickle=True)["data"]
        # raw_data4 = np.load('/SSD2/DY/VadCLIP/exp_ucf/pseudo4/pseudo2_nor_ucf.npz', allow_pickle=True)["data"]
       
    
   
      
        if normal == True and test_mode == False:
            self.df = self.df.loc[self.df['label'] == 'Normal']     # normal 학습 데이터들의 인덱스를 모두 가져옴옴
            self.df = self.df.reset_index()                         # normal 학습 데이터들의 인덱스들을 모두 리셋셋
            # self.pseudo_dict1 = {int(item["index"]): item["pseudo"] for item in raw_data2}
            # self.pseudo_dict2 = {int(item["index"]): item["pseudo"] for item in raw_data4}
            
        elif test_mode == False:
            self.df = self.df.loc[self.df['label'] != 'Normal']     # To get only abnormal data
            self.df = self.df.reset_index()                         # Reset index to start from 0
            # self.pseudo_dict1 = {int(item["index"]): item["pseudo"] for item in raw_data1}
            # self.pseudo_dict2 = {int(item["index"]): item["pseudo"] for item in raw_data3}

    
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        name=[]
        clip_feature = np.load(self.df.loc[index]['path'])
        name.append(self.df.loc[index]['path'])
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        
        # def to_fix_len(arr, T=256, pad_val=-1.0):
        #    if arr is None:
        #       return torch.full((T,), pad_val, dtype=torch.float32)
        #    t = torch.tensor(arr, dtype=torch.float32)
        #    if t.numel() < T:
      
        #       pad_len = T - t.numel()
        #       t = F.pad(t, (0, pad_len), value=pad_val)
        #    elif t.numel() > T:
        #         t = t[:T]
        #    return t
        
        # if self.test_mode == False:
        #     pseudo1 = self.pseudo_dict1.get(index, None)
        #     pseudo2 = self.pseudo_dict2.get(index, None)
        #     pseudo1 = to_fix_len(pseudo1)
        #     pseudo2 = to_fix_len(pseudo2)
            
        #     return clip_feature, clip_label, clip_length  #,index , pseudo1, pseudo2
        
        # else:
        return clip_feature, clip_label, clip_length,name
            
        
        

        
    

class XDDataset(data.Dataset):
    def __init__(self, clip_dim: int, file_path: str, test_mode: bool, label_map: dict):
        self.df = pd.read_csv(file_path)  # 39540
        self.clip_dim = clip_dim
        self.test_mode = test_mode
        self.label_map = label_map
        # raw_data1 = np.load('/SSD2/DY/VadCLIP/exp_xd/pseudo/pseudo1_xd.npz', allow_pickle=True)["data"]
        # raw_data2 = np.load('/SSD2/DY/VadCLIP/exp_xd/pseudo/pseudo2_xd.npz', allow_pickle=True)["data"]
        

        # self.pseudo_dict1 = {int(item["index"]): item["pseudo"] for item in raw_data1}
        # self.pseudo_dict2 = {int(item["index"]): item["pseudo"] for item in raw_data2}

        
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        

        # name=[]
        clip_feature = np.load(self.df.loc[index]['path'])
        name = os.path.splitext(os.path.basename(self.df.loc[index]['path']))[0]
        if self.test_mode == False:
            clip_feature, clip_length = tools.process_feat(clip_feature, self.clip_dim)
        else:
            clip_feature, clip_length = tools.process_split(clip_feature, self.clip_dim)

        clip_feature = torch.tensor(clip_feature)
        clip_label = self.df.loc[index]['label']
        
        def to_fix_len(arr, T=256, pad_val=-1.0):
           if arr is None:
              return torch.full((T,), pad_val, dtype=torch.float32)
           t = torch.tensor(arr, dtype=torch.float32)
           if t.numel() < T:
      
              pad_len = T - t.numel()
              t = F.pad(t, (0, pad_len), value=pad_val)
           elif t.numel() > T:
                t = t[:T]
           return t
        
        if self.test_mode == False:
            return clip_feature, clip_label, clip_length
        else:
            return clip_feature, clip_label, clip_length, name
            
       