import torch
from torch.utils.data import Dataset



class DLoader(Dataset):
    def __init__(self, data, max_seq):
        self.data = data
        self.max_seq = max_seq
        self.data_keys = list(self.data.keys())
        self.length = len(self.data_keys)


    def __getitem__(self, idx):
        key = self.data_keys[idx]
        baseInfo, charteventInfo, label = self.data[key]['baseInfo'], self.data[key]['itemid'], self.data[key]['label']
        try:
            padding = torch.zeros(self.max_seq - charteventInfo.size(0), charteventInfo.size(1))
            charteventInfo = torch.cat((charteventInfo, padding), dim=0)
        except:
            charteventInfo = charteventInfo[:self.max_seq, :]
        return baseInfo, charteventInfo, torch.tensor(label, dtype=torch.float)


    def __len__(self):
        return self.length