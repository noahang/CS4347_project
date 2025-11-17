import torch


class Hparams:
    # Max (and also most) shape from dataset: (237, 84, 201)
    args_2s = {
        'save_model_dir': './Model2/src/results/2s',
        'data_dir': './Model2/data/2s-0.5s/splits',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 1,
        'epoch': 10,
        'lr': 1e-4,
        'num_workers': 0,

        'in_chs': 1,
        'conv1_chs':  64,
        'conv1_ker': (4, 24),
        'stride1': (1, 2),
        'dropout1_prob': 0.05,
        'max_pool_ker1': (1, 2),
        'max_pool_stride1': (1, 2),

        'conv2_chs': 96,
        'conv2_ker': (3, 7),
        'stride2': (2, 2),
        'dropout2_prob': 0.02,
        'max_pool_ker2': (1, 2),
        'max_pool_stride2': (1, 2),

        'conv3_chs': 128,
        'conv3_ker': (3, 5),
        'stride3': (2, 1),
        'dropout3_prob': 0.05,

        'lstm_chs': 2432,       # This param may be incorrect
        'lstm_hidden': 256,

        'fc_chs': 128,
        'out_chs': 84,
    }

    # Max (and also most) shape from dataset: (115, 84, 601)
    args_6s = {
        'save_model_dir': './Model2/src/results/6s',
        'data_dir': './Model2/data/6s-1s/splits',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 1,        # Consider to change
        'epoch': 10,
        'lr': 1e-4,
        'num_workers': 0,

        'in_chs': 1,
        'conv1_chs': 64,
        'conv1_ker': (4, 24),
        'stride1': (1, 2),
        'dropout1_prob': 0.05,
        'max_pool_ker1': (1, 2),
        'max_pool_stride1': (1, 2),

        'conv2_chs': 96,
        'conv2_ker': (3, 7),
        'stride2': (2, 2),
        'dropout2_prob': 0.02,
        'max_pool_ker2': (1, 2),
        'max_pool_stride2': (1, 2),

        'conv3_chs': 128,
        'conv3_ker': (3, 5),
        'stride3': (2, 1),
        'dropout3_prob': 0.05,

        'lstm_chs': 72960,
        'lstm_hidden': 256,

        'fc_chs': 128,

        'tonal_center_out_chs': 12,
        'mode_out_chs': 7,
    }
