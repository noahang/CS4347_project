import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import time
import pickle
import argparse
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

from Model2.src.models import CnnLstm
from Model2.src.hparams import Hparams

from Model2.src.dataset import get_data_loader, move_data_to_device
from Model2.src.config.mode_map import NUM_TO_SCALE_MAP, NUM_TO_TONAL_CENTER_MAP, NUM_TO_MUSICAL_MODE_MAP


# from utils import ls
# import matplotlib.pyplot as plt

# import warnings
#
# warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = '3' # If you have multiple GPU's,
# uncomment this line to specify which GPU you want to use


def main():
    classifier = Classifier(device=Hparams.args_6s['device'])

    best_model_id = classifier.fit(Hparams.args_6s)
    print("Best model from epoch: ", best_model_id)


class Classifier:
    def __init__(self, device="cpu", model_path=None, model_type='6s'):
        self.device = device
        if model_type == '2s':
            self.model = CnnLstm(Hparams.args_2s).to(self.device)
        else:
            self.model = CnnLstm(Hparams.args_6s).to(self.device)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print('Model loaded.')
        else:
            print('Model initialized.')

    def fit(self, args):
        # Set paths
        save_model_dir = args['save_model_dir']
        if not os.path.exists(save_model_dir):
            os.mkdir(save_model_dir)

        optimizer = optim.AdamW(self.model.parameters(), lr=args['lr'])
        loss_func = nn.CrossEntropyLoss()
        metric = Metrics(loss_func)

        train_loader = get_data_loader(split='train', args=args)
        valid_loader = get_data_loader(split='valid', args=args)

        # Start training
        print('Start training...')
        start_time = time.time()
        best_model_id = -1
        min_valid_loss = 10000

        for epoch in range(1, args['epoch'] + 1):
            self.model.train()
            total_training_loss = 0
            train_outs = []
            train_tgts = []

            # Train
            pbar = tqdm(train_loader)
            for i, batch in enumerate(pbar):
                x, tgt_center, tgt_mode = move_data_to_device(batch, self.device)
                # print(f"x.shape: {x.shape}, center.shape: {tgt_center.shape}, mode.shape: {tgt_mode.shape}")
                out = self.model(x)
                # print(1)
                loss_center = loss_func(out[0], tgt_center)
                loss_mode = loss_func(out[1], tgt_mode)
                losses = (loss_center + loss_mode, loss_center, loss_mode)
                loss = losses[0]
                metric.update(out, (tgt_center, tgt_mode), losses)
                # print(2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item()
                # print(3)

                pbar.set_description('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))
            metric_train = metric.get_value()

            # Validation
            self.model.eval()
            with torch.no_grad():
                for batch in valid_loader:
                    x, tgt_center, tgt_mode = move_data_to_device(batch, args['device'])
                    out = self.model(x)
                    metric.update(out, (tgt_center, tgt_mode))
            metric_valid = metric.get_value()

            # Logging
            print('[Epoch {:02d}], Train Loss: {:.5f}, Valid Loss {:.5f}, Time {:.2f}s'.format(
                epoch, metric_train['loss'], metric_valid['loss'], time.time() - start_time,
            ))
            print('Split Train ACC: Tonal Center: {:.4f}, Musical Mode: {:.4f}'.format(
                metric_train['center_acc'],
                metric_train['mode_acc']
            ))
            print('Split Valid Acc: Tonal Center: {:.4f}, Musical Mode: {:.4f}'.format(
                metric_valid['center_acc'],
                metric_valid['mode_acc']
            ))
            print('Split Train Loss: Tonal Center: {:.4f}, Musical Mode: {:.4f}'.format(
                metric_train['center_loss'],
                metric_train['mode_loss']
            ))
            print('Split Valid Loss: Tonal Center: {:.4f}, Musical Mode: {:.4f}'.format(
                metric_valid['center_loss'],
                metric_valid['mode_loss']
            ))

            # Save the best model
            if metric_valid['loss'] < min_valid_loss:
                min_valid_loss = metric_valid['loss']
                best_model_id = epoch

                save_dict = self.model.state_dict()
                target_model_path = save_model_dir + '/best_model3.pth'
                torch.save(save_dict, target_model_path)

        print('Training done in {:.1f} minutes.'.format((time.time() - start_time) / 60))
        return best_model_id

    def classify(self, features: np.ndarray) -> str:
        int_center, int_mode = self.model(features)
        return f"{NUM_TO_TONAL_CENTER_MAP[int_center]} {NUM_TO_MUSICAL_MODE_MAP[int_mode]}"


class Metrics:
    def __init__(self, loss_func):
        self.buffer = {}
        self.loss_func = loss_func

    def update(self, out, tgt, losses=None):
        with torch.no_grad():
            out_center, out_mode = out
            tgt_center, tgt_mode = tgt

            if losses == None:
                loss_center = self.loss_func(out[0], tgt_center)
                loss_mode = self.loss_func(out[1], tgt_mode)
                losses = (loss_center + loss_mode, loss_center, loss_mode)

            pred_center = torch.argmax(out_center, dim=-1)
            pred_mode = torch.argmax(out_mode, dim=-1)

            acc_center = self.get_acc(pred_center, tgt_center)
            acc_mode = self.get_acc(pred_mode, tgt_mode)

            batch_metric = {
                'loss': losses[0].item(),
                'center_loss': losses[1].item(),
                'mode_loss': losses[2].item(),
                'center_acc': acc_center,
                'mode_acc': acc_mode,
            }

            for k in batch_metric:
                if k in self.buffer:
                    self.buffer[k].append(batch_metric[k])
                else:
                    self.buffer[k] = [batch_metric[k]]

    def get_acc(self, out, tgt):
        out = torch.flatten(out)
        tgt = torch.flatten(tgt).float()

        return torch.mean((tgt == out).float())

    def get_value(self):
            for k in self.buffer:
                self.buffer[k] = sum(self.buffer[k]) / len(self.buffer[k])
            ret = self.buffer
            self.buffer = {}
            return ret


if __name__ == '__main__':
    main()
