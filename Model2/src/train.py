import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import time
from matplotlib import pyplot as plt
from tqdm import tqdm

from Model2.src.config.mode_map import NUM_TO_TONAL_CENTER_MAP, NUM_TO_MUSICAL_MODE_MAP, get_scales_from_nums
from Model2.src.dataset import get_data_loader, move_data_to_device
from Model2.src.hparams import Hparams
from Model2.src.models import CnnLstm

import warnings

warnings.filterwarnings('ignore')


def main():
    classifier = Classifier(device=Hparams.args_6s['device'])

    best_model_id = classifier.fit(Hparams.args_6s)
    print("Best model from epoch: ", best_model_id)


class Classifier:
    def __init__(self, device="cpu", model_path=None, model_type='6s'):
        self.device = device
        if model_type == '2s':
            self.model = CnnLstm(Hparams.args_2s).to(self.device)
        elif model_type == '6s':
            self.model = CnnLstm(Hparams.args_6s).to(self.device)
        else:
            raise ValueError("Model type must be '2s' or '6s'")

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

            # Train
            print(f"Training...")
            pbar = tqdm(train_loader)
            for i, batch in enumerate(pbar):
                x, tgt_center, tgt_mode = move_data_to_device(batch, self.device)
                print(f"Song {i}, tgt={get_scales_from_nums(tgt_center, tgt_mode)[0]}")
                out = self.model(x)
                loss_center = loss_func(out[0], tgt_center)
                loss_mode = loss_func(out[1], tgt_mode)
                losses = (loss_center + loss_mode, loss_center, loss_mode)
                loss = losses[0]
                metric.update(out, (tgt_center, tgt_mode), losses)
                print(f"        out={metric.get_pred(out)}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_training_loss += loss.item()

                pbar.set_description('Epoch {}, Loss: {:.4f}'.format(epoch, loss.item()))
            metric_train = metric.get_value()

            # Validation
            # print("Validating")
            self.model.eval()
            with torch.no_grad():
                for i, batch in enumerate(valid_loader):
                    x, tgt_center, tgt_mode = move_data_to_device(batch, args['device'])
                    # print(f"Song {i}, tgt={get_scale_from_nums(tgt_center, tgt_mode)}")
                    out = self.model(x)
                    # print(f"        out={get_scale_from_nums(out[0], out[1])}")
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

            metric.plot_metrics(save_path=save_model_dir + f"/imgs/metrics_plot{epoch}.png")

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
        self.train_losses = []
        self.valid_losses = []
        self.train_accs = []
        self.valid_accs = []

    def update(self, out, tgt, losses=None):
        with torch.no_grad():
            store_mode = 0
            out_center, out_mode = out
            tgt_center, tgt_mode = tgt

            if losses == None:
                store_mode = 1
                loss_center = self.loss_func(out[0], tgt_center)
                loss_mode = self.loss_func(out[1], tgt_mode)
                losses = (loss_center + loss_mode, loss_center, loss_mode)

            pred_center = torch.argmax(out_center, dim=-1)
            pred_mode = torch.argmax(out_mode, dim=-1)

            acc_center = self.get_acc(pred_center, tgt_center)
            acc_mode = self.get_acc(pred_mode, tgt_mode)

            if store_mode == 0:
                self.train_losses.append((
                    losses[0].detach().numpy(),
                    losses[1].detach().numpy(),
                    losses[2].detach().numpy()
                ))
                self.train_accs.append((
                    acc_center.detach().numpy(),
                    acc_mode.detach().numpy()
                ))
            else:
                self.valid_losses.append((
                    losses[0].detach().numpy(),
                    losses[1].detach().numpy(),
                    losses[2].detach().numpy()
                ))
                self.valid_accs.append((
                    acc_center.detach().numpy(),
                    acc_mode.detach().numpy()
                ))

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

    def plot_metrics(self, save_path="training_curves.png"):
        train_losses = np.array(self.train_losses)
        valid_losses = np.array(self.valid_losses)
        train_accs = np.array(self.train_accs)
        valid_accs = np.array(self.valid_accs)

        train_total_loss = train_losses[:, 0]
        train_center_loss = train_losses[:, 1]
        train_mode_loss = train_losses[:, 2]

        valid_total_loss = valid_losses[:, 0]
        valid_center_loss = valid_losses[:, 1]
        valid_mode_loss = valid_losses[:, 2]

        train_center_acc = train_accs[:, 0]
        train_mode_acc = train_accs[:, 1]

        valid_center_acc = valid_accs[:, 0]
        valid_mode_acc = valid_accs[:, 1]

        x1 = np.arange(1, len(train_losses) + 1)
        x2 = np.arange(1, len(valid_losses) + 1)

        # ------------------- Start Plotting -------------------
        plt.figure(figsize=(14, 10))

        # ---- Loss Curves ----
        plt.subplot(2, 1, 1)
        plt.title("Loss per Epoch")

        plt.plot(x1, train_total_loss, label="Train Total Loss")
        plt.plot(x2, valid_total_loss, label="Valid Total Loss")

        plt.plot(x1, train_center_loss, "--", label="Train Center Loss")
        plt.plot(x2, valid_center_loss, "--", label="Valid Center Loss")

        plt.plot(x1, train_mode_loss, ":", label="Train Mode Loss")
        plt.plot(x2, valid_mode_loss, ":", label="Valid Mode Loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # ---- Accuracy Curves ----
        plt.subplot(2, 1, 2)
        plt.title("Accuracy per Epoch")

        plt.plot(x1, train_center_acc, label="Train Center Acc")
        plt.plot(x2, valid_center_acc, label="Valid Center Acc")
        plt.plot(x1, train_mode_acc, "--", label="Train Mode Acc")
        plt.plot(x2, valid_mode_acc, "--", label="Valid Mode Acc")

        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

        print(f"Training curves saved to {save_path}")

    def get_pred(self, out):
        with torch.no_grad():
            out_center, out_mode = out

            pred_center = torch.argmax(out_center, dim=-1)
            pred_mode = torch.argmax(out_mode, dim=-1)

            return get_scales_from_nums(pred_center, pred_mode)


if __name__ == '__main__':
    main()
