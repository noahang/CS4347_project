import torch
import torch.nn as nn

from Model2.src.hparams import Hparams


class CnnLstm(nn.Module):
    def __init__(self, args: Hparams.args_6s):
        super(CnnLstm, self).__init__()
        self.args = args
        self.device = args['device']

        self.conv1 = nn.Sequential(
            nn.Conv2d(args['in_chs'], args['conv1_chs'], kernel_size=args['conv1_ker'],
                      # padding=args['padding'],
                      stride=args['stride1']),
            nn.BatchNorm2d(args['conv1_chs']),
            nn.ReLU(),
            nn.Dropout2d(p=args['dropout1_prob']),
            nn.MaxPool2d(kernel_size=args['max_pool_ker1'], stride=args['max_pool_stride1']),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(args['conv1_chs'], args['conv2_chs'], kernel_size=args['conv2_ker'],
                      # padding=args['padding'],
                      stride=args['stride2']),
            nn.BatchNorm2d(args['conv2_chs']),
            nn.ReLU(),
            nn.Dropout2d(p=args['dropout2_prob']),
            nn.MaxPool2d(kernel_size=args['max_pool_ker2'], stride=args['max_pool_stride2']),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(args['conv2_chs'], args['conv3_chs'], kernel_size=args['conv3_ker'],
                      # padding=args['padding'],
                      stride=args['stride3']),
            nn.BatchNorm2d(args['conv3_chs']),
            nn.ReLU(),
            nn.Dropout2d(p=args['dropout3_prob']),
        )

        self.lstm = nn.LSTM(
            input_size=args['lstm_chs'],
            hidden_size=args['lstm_hidden'],
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(args['lstm_hidden'], args['fc_chs']),
            nn.ReLU(),
        )
        self.tonal_center_head = nn.Linear(args['fc_chs'], args['tonal_center_out_chs'])
        self.musical_mode_head = nn.Linear(args['fc_chs'], args['mode_out_chs'])

    def forward(self, x):
        # B (num of 6s for 1 song), Cin (1), F (frequency=84), T (time=601)
        # Cout (128)
        x = x.unsqueeze(1)          # (B, Cin, F, T)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)         # (B, Cout, F, T')

        c3 = c3.flatten(1)          # (115, 72960)
        c3 = c3.unsqueeze(1)

        h0 = torch.zeros(1, c3.shape[0], self.args['lstm_hidden']).to(self.device)
        c0 = torch.zeros(1, c3.shape[0], self.args['lstm_hidden']).to(self.device)
        lstm1, _ = self.lstm(c3, (h0, c0))              # (115, 1, 256)

        f1 = self.fc(lstm1[:, -1, :])                   # (115, 128)

        out_tonal_center = self.tonal_center_head(f1)   # (115, 12)
        out_musical_mode = self.musical_mode_head(f1)   # (115, 8)

        return out_tonal_center, out_musical_mode
