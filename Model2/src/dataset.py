import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config.mode_map import MUSICAL_MODE_TO_NUM_MAP, TONAL_CENTER_TO_NUM_MAP, SCALE_TO_NUM_MAP


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

            scales = [d for d in os.listdir(self.ds_pth)
                      if os.path.isdir(os.path.join(self.ds_pth, d)) and d in SCALE_TO_NUM_MAP.keys()]

            for scale in scales:
                folder_pth = os.path.join(self.ds_pth, scale)
                center, mode = scale.split(" ")
                center_idx = TONAL_CENTER_TO_NUM_MAP[center]
                mode_idx = MUSICAL_MODE_TO_NUM_MAP[mode]
                for f in os.listdir(folder_pth):
                    if f[0] == '.':
                        continue
                    f_pth = os.path.join(folder_pth, f)
                    data = np.load(f_pth).astype(np.float32)
                    self.x.append(torch.tensor(data))
                    self.tgt.append((torch.tensor([center_idx]), torch.tensor([mode_idx])))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.tgt[idx][0], self.tgt[idx][1]


def get_data_loader(split, args):
    ds = MyDataset(
        ds_root=args['data_dir'],
        split=split
    )
    data_loader = DataLoader(
        ds,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
        collate_fn=collate_fn
    )
    return data_loader



def move_data_to_device(data, device):
    ret = []
    for i in data:
        if isinstance(i, torch.Tensor):
            ret.append(i.to(device))
    return ret


def collate_fn(batch):
    inp = []
    tgt_center = []
    tgt_mode = []
    max_frame_num = max(sample[0].shape[0] for sample in batch)

    for x, y_center, y_mode in batch:
        if x.shape[0] < max_frame_num:
            pad_amount = max_frame_num - x.shape[0]
            x = torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_amount), mode='constant', value=0)
        y_center = torch.nn.functional.pad(y_center, (0, max_frame_num - y_center.shape[0]), mode='constant', value=y_center.item())
        y_mode = torch.nn.functional.pad(y_mode, (0, max_frame_num - y_mode.shape[0]), mode='constant', value=y_mode.item())
        inp.append(x)
        tgt_center.append(y_center)
        tgt_mode.append(y_mode)

    inp = torch.stack(inp).squeeze(0)
    tgt_center = torch.stack(tgt_center).squeeze(0)
    tgt_mode = torch.stack(tgt_mode).squeeze(0)
    return inp, tgt_center, tgt_mode
