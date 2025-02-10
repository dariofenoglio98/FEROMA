# ANDA interface

import config as cfg

import numpy as np
import argparse
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from ANDA import anda

# Get arguments
parser = argparse.ArgumentParser(description='Generate datasets for ANDA')
parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
args = parser.parse_args()

# valid dataset names
assert cfg.dataset_name in ['CIFAR10', 'CIFAR100', 'MNIST', 'FMNIST', 'EMNIST'], \
        "Dataset not found! Please check the ANDA page for more details."

# Create folder if not exist
os.makedirs('./data/cur_datasets', exist_ok=True)

anda_dataset = []

# special static mode using unique fn
if cfg.drifting_type == 'static':
    assert cfg.non_iid_type in ['feature_skew',
                                'label_skew',
                                'feature_condition_skew',
                                'label_condition_skew',
                                'split_unbalanced',
                                'feature_label_skew',
                                'feature_condition_skew_with_label_skew',
                                'label_condition_skew_with_label_skew',
                                'label_condition_skew_unbalanced',
                                'feature_condition_skew_unbalanced',
                                'label_skew_unbalanced',
                                'feature_skew_unbalanced',
                                'feature_skew_strict',
                                'label_skew_strict',
                                'feature_condition_skew_strict',
                                'label_condition_skew_strict',
    ], "Non-IID type not supported in static mode! Please check the ANDA page for more details."
    anda_dataset = anda.load_split_datasets(
        dataset_name = cfg.dataset_name,
        client_number = cfg.n_clients,
        non_iid_type = cfg.non_iid_type,
        mode = "manual",
        verbose = cfg.verbose,
        count_labels=cfg.count_labels,
        plot_clients=cfg.plot_clients,
        random_seed = cfg.random_seed + args.fold,
        **cfg.args
    )
elif cfg.drifting_type in ['trND_teDR','trDA_teDR','trDA_teND','trDR_teDR','trDR_teND']:
    # dynamic mode using same fn
    anda_dataset = anda.load_split_datasets_dynamic(
        dataset_name = cfg.dataset_name,
        client_number = cfg.n_clients,
        non_iid_type = cfg.non_iid_type,
        drfting_type = cfg.drifting_type,
        verbose = cfg.verbose,
        count_labels=cfg.count_labels,
        plot_clients=cfg.plot_clients,
        random_seed = cfg.random_seed + args.fold,
        **cfg.args
    )
else:
    raise ValueError("Drifting type not found! Please check the ANDA page for more details.")

# Save anda_dataset
# simple format as training not drifting
if not cfg.training_drifting:
    n_clusters = []
    for client_number in range(cfg.n_clients):
        np.save(f'./data/cur_datasets/client_{client_number}', anda_dataset[client_number])
        print(f"Data for client {client_number} saved")
        n_clusters.append(anda_dataset[client_number]["cluster"])
    n_clusters = np.unique(n_clusters).shape[0]
    print(f"\033[91mNumber of correct clusters: {n_clusters}\033[0m")
    np.save(f'./data/cur_datasets/n_clusters.npy', n_clusters)

# complex format as training drifting
else:
    drifting_log = {}
    for dataset in anda_dataset:
        client_number = dataset['client_number']
        cur_drifting_round = int(cfg.n_rounds * dataset['epoch_locker_indicator']) if dataset['epoch_locker_indicator'] != -1 else -1

        # save data file      
        filename = f'./data/cur_datasets/client_{client_number}_round_{cur_drifting_round}.npy'
        np.save(filename, dataset)
        print(f"Data for client {client_number} round {cur_drifting_round} saved")

        # log drifting round info
        if client_number not in drifting_log:
            drifting_log[client_number] = []
        drifting_log[client_number].append(cur_drifting_round)

    # print(", ".join(f"{key}: {value}" for key, value in drifting_log.items()))

    # save log file
    np.save(f'./data/cur_datasets/drifting_log.npy', drifting_log)

print("Datasets saved successfully!")

