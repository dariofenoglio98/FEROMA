# Overall settings
k_folds = 5 # number of folds for cross-validation, if 1, no cross-validation
strategy = 'cfl_oneshot' # ['fedavg', 'fedprox', 'cfl_oneshot', 'cfl_drift', 'optimal_FL']
random_seed = 42
gpu = -2 # set the GPU to use, if -1 use CPU, -2 for multigpus
n_clients = 10
n_samples_clients = -1 # if -1, use all samples

# differential privacy on the descriptors
differential_privacy_descriptors = True
epsilon = 1.0
# sensitivity = 1.0 # automatically calculated

# Strategy cfl_oneshot
cfl_oneshot_CLIENT_SCALING_METHOD = 1 # ['Ours', 'weighted', 'none']
cfl_oneshot_CLIENT_CLUSTER_METHOD = 4 # ['Kmeans', 'DBSCAN', 'HDBSCAN', 'DBSCAN_no_outliers', 'Kmeans_with_prior']
extended_descriptors = True #mean and std 
weighted_metric_descriptors = False
selected_descriptors = "Px_label_long" # Options: "Px", "Py", "Pxy", "Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short" for training time
pos_multiplier = 6 # positional embedding multiplier 
# check_cluster_at_inference = False ALWAYS BOTH  # True if you want to check the cluster at inference time (test-time inference for test drifting-find closest cluster to you), False otherwise (like baselines)
eps_scaling = 1.0 # for clustering method 4
th_round = 0.06 # derivative threshold on accuracy trend for starting clustering (good enough evaluation model)

# Strategy fedprox
fedprox_proximal_mu = 0.1

# Dataset settings
dataset_name = "MNIST" # ["CIFAR10", "CIFAR100", "MNIST", "FMNIST", "EMNIST"]
drifting_type = 'static' # ['static', 'trND_teDR', 'trDA_teDR', 'trDA_teND', 'trDR_teDR', 'trDR_teND'] refer to ANDA page for more details
non_iid_type = 'feature_skew_strict' # ['feature_skew_strict', 'label_skew_strict', 'feature_condition_skew', 'label_condition_skew'] refer to ANDA page for more details
verbose = True
count_labels = True
plot_clients = False

# Careful with the args applying to your settings above
# # FEATURE DISTRIBUTION SHIFT P(X) - (feature_skew_strict) 
args = {
    'set_rotation': True,
    'set_color': True,
    'rotations':4,
    'colors':1,
}

# # LABEL DISTRIBUTION SHIFT P(Y) - (label_skew_strict)
# args = {
#     'py_bank': 5,
#     'client_n_class': 5,
# }

# # CONCEPT DRIFT P(Y|X) - (feature_condition_skew)
# args = {
#     'random_mode':True,
#     'mixing_label_number':7, # was 3 for new_config
#     'scaling_label_low':1.0,
#     'scaling_label_high':1.0,
# }

# # CONCEPT DRIFT P(X|Y) - (label_condition_skew)
# args = {
#         'set_rotation': True,
#         'set_color': True,
#         'rotations':4,
#         'colors':1,
#         'random_mode':True,
#         'rotated_label_number':1,
#         'colored_label_number':1,
# }

# Training model settings
model_name = "LeNet5"   # ["LeNet5", "ResNet9"]
batch_size = 64
test_batch_size = 64
client_eval_ratio = 0.2
n_rounds = 10
local_epochs = 2
lr = 0.005
momentum = 0.9

# self-defined settings
n_classes_dict = {
    "CIFAR10": 10,
    "CIFAR100": 100,
    "MNIST": 10,
    "FMNIST": 10
}
n_classes = n_classes_dict[dataset_name]

input_size_dict = {
    "CIFAR10": (32, 32),
    "CIFAR100": (32, 32),
    "MNIST": (28, 28),
    "FMNIST": (28, 28)
}
input_size = input_size_dict[dataset_name]

acceptable_accuracy = {
    "CIFAR10": 0.5,
    "CIFAR100": 0.1,
    "MNIST": 0.8,
    "FMNIST": 0.8
}
th_accuracy = acceptable_accuracy[dataset_name]
training_drifting = False if drifting_type in ['static', 'trND_teDR'] else True # to be identified
default_path = f"{random_seed}/{model_name}/{dataset_name}/{drifting_type}"

# FL settings - Communications
port = '8018'
ip = '0.0.0.0' # Local Host=0.0.0.0, or IP address of the server

# Advance One-shot settings
len_metric_descriptor =  n_classes
n_metrics_descriptors = 2 if extended_descriptors else 1
len_latent_space_descriptor = 1 * len_metric_descriptor   # modify this to change the latent space size
n_latent_space_descriptors = 2 if extended_descriptors else 1
