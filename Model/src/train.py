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

from models import CnnLstm
from hparams import Hparams

from dataset import get_data_loaders, move_data_to_device
from config.mode_map import NUM_TO_MODE_MAP


# from utils import ls
# import matplotlib.pyplot as plt

# import warnings
#
# warnings.filterwarnings('ignore')

# os.environ["CUDA_VISIBLE_DEVICES"] = '3' # If you have multiple GPU's,
# uncomment this line to specify which GPU you want to use


def main():
    classifier = Classifier(device=Hparams.args_2s['device'])

    best_model_id = classifier.fit(Hparams.args_2s)
    print("Best model from epoch: ", best_model_id)


class Classifier:
    def __init__(self, device="cpu", model_path=None, model_type='2s'):
        self.device = device
        if model_type == '2s':
            self.model = CnnLstm(Hparams.args_2s).to(self.device)
        # else:
        #     self.model = CnnLSTM(Hparams.args_2s).to(self.device)

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

        # ? adam with adam2
        optimizer = optim.AdamW(self.model.parameters(), lr=args['lr'])
        loss_func = nn.CrossEntropyLoss()

        train_loader, valid_loader = get_data_loaders(split='train', args=args)

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
                x, tgt = move_data_to_device(batch, self.device)
                # tgts = torch.nn.functional.one_hot(tgt, 84).float()
                out = self.model(x)
                loss = loss_func(out, tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item()
                pbar.set_description('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))

                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)
                train_outs.extend(preds.cpu().numpy().tolist())
                train_tgts.extend(tgt.cpu().numpy().tolist())

            print(f"Epoch {epoch}: Loss: {total_training_loss}")
            train_avg_loss = total_training_loss / len(train_loader)
            train_acc = (np.array(train_outs) == np.array(train_tgts)).mean()
            # train_f1 = f1_score(train_outs, train_tgts, average='macro')

            # Validation
            self.model.eval()
            total_valid_loss = 0.0
            valid_outs = []
            valid_tgts = []
            with torch.no_grad():
                for batch in valid_loader:
                    x, tgt = move_data_to_device(batch, args['device'])
                    # tgts = torch.nn.functional.one_hot(tgt, 84).long()
                    out = self.model(x)
                    loss = loss_func(out, tgt)
                    total_valid_loss += loss.item()

                    probs = torch.softmax(out, dim=1)
                    preds = torch.argmax(probs, dim=1)
                    valid_outs.extend(preds.cpu().numpy().tolist())
                    valid_tgts.extend(tgt.cpu().numpy().tolist())

            valid_avg_loss = total_valid_loss / len(valid_loader)
            valid_acc = (np.array(valid_outs) == np.array(valid_tgts)).mean()
            # valid_f1 = f1_score(valid_outs, valid_tgts, average='macro')

            # Logging
            print('[Epoch {:02d}], Train Loss: {:.5f}, Valid Loss {:.5f}, Time {:.2f}s'.format(
                epoch, train_avg_loss, valid_avg_loss, time.time() - start_time,
            ))
            print('Split Train Acc: {:.4f}'.format(
                train_acc,
                # train_f1,
            ))
            print('Split Valid Acc: {:.4f}'.format(
                valid_acc,
                # valid_f1,
            ))

            # Save the best model
            if valid_avg_loss < min_valid_loss:
                min_valid_loss = valid_avg_loss
                best_model_id = epoch

                save_dict = self.model.state_dict()
                target_model_path = save_model_dir + '/best_model2.pth'
                torch.save(save_dict, target_model_path)

        print('Training done in {:.1f} minutes.'.format((time.time() - start_time) / 60))
        return best_model_id

    def classify(self, features: np.ndarray) -> str:
        int_mode = self.model(features)
        return NUM_TO_MODE_MAP[int_mode]


if __name__ == '__main__':
    main()
