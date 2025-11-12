import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config.mode_map import MODE_TO_NUM_MAP


class MyDataset(Dataset):
    def __init__(self, ds_root=None, split=None, x=None, tgt=None):
        if x is not None and tgt is not None:
            self.x = x
            self.tgt = tgt

        else:
            self.ds_root = ds_root
            self.split = split
            self.ds_pth = os.path.join(ds_root, split)

            self.x = []
            self.tgt = []

            modes = [d for d in os.listdir(self.ds_pth)
                     if os.path.isdir(os.path.join(self.ds_pth, d)) and d in MODE_TO_NUM_MAP.keys()]

            for mode in modes:
                folder_pth = os.path.join(self.ds_pth, mode)
                idx = MODE_TO_NUM_MAP[mode]
                for f in os.listdir(folder_pth):
                    f_pth = os.path.join(folder_pth, f)
                    data = np.load(f_pth).astype(np.float32)
                    self.x.append(torch.tensor(data))
                    self.tgt.append(torch.tensor([idx]))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.tgt[idx]


def get_data_loaders(split, args):
    ds = MyDataset(ds_root=args['data_dir'], split=split)
    if split == 'test':
        data_loader = DataLoader(ds, batch_size=args['batch_size'], shuffle=False,
                                 num_workers=args['num_workers'])
        return data_loader
    elif split == 'train':
        n = len(ds)
        num_valid = int(n * 0.2)
        num_train = n - num_valid

        # train_ds = MyDataset(x=ds.x[:num_train], tgt=ds.tgt[:num_train])
        # valid_ds = MyDataset(x=ds.x[num_train:], tgt=ds.tgt[num_train:])
        train_ds = MyDataset(x=ds.x[:800], tgt=ds.tgt[:800])
        valid_ds = MyDataset(x=ds.x[num_train:num_train+200], tgt=ds.tgt[num_train:num_train+200])

        train_data_loader = DataLoader(train_ds, batch_size=args['batch_size'], shuffle=False,
                                       num_workers=args['num_workers'], collate_fn=collate_fn,)
        valid_data_loader = DataLoader(valid_ds, batch_size=args['batch_size'], shuffle=False,
                                       num_workers=args['num_workers'], collate_fn=collate_fn,)
        return train_data_loader, valid_data_loader


def move_data_to_device(data, device):
    ret = []
    for i in data:
        if isinstance(i, torch.Tensor):
            ret.append(i.to(device))
    return ret


def collate_fn(batch):
    inp = []
    tgt = []
    max_frame_num = max(sample[0].shape[0] for sample in batch)
    F = batch[0][0].shape[1]
    T = batch[0][0].shape[2]

    for x, y in batch:
        if x.shape[0] < max_frame_num:
            pad_amount = max_frame_num - x.shape[0]
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_amount), mode='constant', value=0)
        y = torch.nn.functional.pad(y, (0, max_frame_num-1), mode='constant', value=y.item())
        inp.append(x)
        tgt.append(y)

    inp = torch.stack(inp).reshape(-1, F, T)
    tgt = torch.stack(tgt).reshape(-1)
    return inp, tgt
