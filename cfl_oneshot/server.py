"""
This code implements the FedAvg, when it starts, the server waits for the clients to connect. When the established number 
of clients is reached, the learning process starts. The server sends the model to the clients, and the clients train the 
model locally. After training, the clients send the updated model back to the server. Then client models are aggregated 
with FedAvg. The aggregated model is then sent to the clients for the next round of training. The server saves the model 
and metrics after each round.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).

METHOD: in the first rounds, FedAvg is used until the global model reaches a pre-defined accuracy. After that the 
current global model is utilized to extract client descriptors and perform the one-shot clustering. After the clustering,
each client receives only the assigned cluster model, which its local model will be aggregated with other client models
in the same clusters. The training continues until the end. 
"""

# Libraries
import json
import copy
import time
import torch
import argparse
import numpy as np
from functools import reduce
from logging import WARNING
from torch.utils.data import DataLoader
from collections import OrderedDict
from typing import List, Tuple, Union, Optional, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from kneed import KneeLocator # type: ignore

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import public.config as cfg
import public.utils as utils
import public.models as models

import flwr as fl
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager
from flwr.common.logger import log
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
    NDArrays,
)

MAX_LATENT_SPACE = 2

# TODO DARIO
# WHAT IF: we introduced latent space descriptors per class, i.e., selecting only one class, calculating the mean latent
# space on it, reducing dim, and then same for others. Creating something like [metrics, latent_class1, latent_class2..]

class client_descr_scaling:
    def __init__(self, 
                 scaling_method: int = 1, 
                 scaler = None, # MinMaxScaler() or StandardScaler()
                 *args,
                 **kwargs):
        self.scaling_method = scaling_method
        self.scaler = scaler
        self.scalers = None
        self.fitted = False 
        if cfg.selected_descriptors == 'Px':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors
            self.num_scalers = cfg.n_latent_space_descriptors
        elif cfg.selected_descriptors == 'Py':
            self.descriptors_dim = [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors 
            self.num_scalers = cfg.n_metrics_descriptors
        elif cfg.selected_descriptors == 'Px_cond':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2
            self.num_scalers = cfg.n_latent_space_descriptors * 2
        elif cfg.selected_descriptors == 'Pxy_cond':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2 + [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors
            self.num_scalers = cfg.n_latent_space_descriptors * 2 + cfg.n_metrics_descriptors
        elif cfg.selected_descriptors == 'Px_label_long':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * (cfg.n_classes + 1)
            self.num_scalers = cfg.n_latent_space_descriptors * (cfg.n_classes + 1)
        elif cfg.selected_descriptors == 'Px_label_short':
            self.descriptors_dim = [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors * 2
            self.num_scalers = cfg.n_latent_space_descriptors * 2
        else:
            self.descriptors_dim = [cfg.len_metric_descriptor] * cfg.n_metrics_descriptors + [cfg.len_latent_space_descriptor] * cfg.n_latent_space_descriptors
            self.num_scalers = cfg.n_metrics_descriptors + cfg.n_latent_space_descriptors

        print(f"n scalers: {self.num_scalers} - desc dim {self.descriptors_dim}")

    def scale(self, client_descr: np.ndarray = None) -> np.ndarray:
        # Normalize by group of descriptors
        if self.scaling_method == 1:
            if self.scalers is None:
                self.scalers = [copy.deepcopy(self.scaler) for _ in range(self.num_scalers)]
                self.dim = client_descr.shape[1]
             
            if self.fitted:
                if client_descr.shape[1] != self.dim:
                    raise ValueError("Client descriptors dimension mismatch!")
                scaled_client_descr = np.zeros(client_descr.shape)
                start_idx = 0
                for i, (scaler, descr_dim) in enumerate(zip(self.scalers, self.descriptors_dim)):
                    end_idx = start_idx + descr_dim
                    single_client_descr = client_descr[:, start_idx:end_idx]
                    scaled_client_descr[:, start_idx:end_idx] = scaler.transform(
                        single_client_descr.reshape(-1, 1)).reshape(single_client_descr.shape)
                    start_idx = end_idx
            else:
                self.fitted = True
                scaled_client_descr = np.zeros(client_descr.shape)
                start_idx = 0
                for i, (scaler, descr_dim) in enumerate(zip(self.scalers, self.descriptors_dim)):
                    end_idx = start_idx + descr_dim
                    single_client_descr = client_descr[:, start_idx:end_idx]
                    scaled_client_descr[:, start_idx:end_idx] = scaler.fit_transform(
                        single_client_descr.reshape(-1, 1)).reshape(single_client_descr.shape)
                    start_idx = end_idx
                
            return scaled_client_descr
        
        elif self.scaling_method == 2:
            # TODO weighted scaling
            return None
        
        elif self.scaling_method == 3:
            # No scaling
            return client_descr
        
        else:
            print("Invalid scaling method!")
            return None


# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": cfg.local_epochs,
        "tot_rounds": cfg.n_rounds,
        "extract_descriptors": False, 
        "min_latent_space": 0,
        "max_latent_space": MAX_LATENT_SPACE,
    }
    return config



# Custom weighted average function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    # validities = [num_examples * m["validity"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}



def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples



def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]

    # Compute average weights of each layer
    weights_prime: NDArrays = [
        reduce(np.add, layer_updates) / num_examples_total
        for layer_updates in zip(*weighted_weights)
    ]
    return weights_prime



# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, path, descriptors_scaler, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model # used for saving checkpoints
        self.path = path # saving model path
        self.descriptors_scaler = descriptors_scaler # used for scaling client descriptors

        self.aggregated_cluster_parameters = {} # [cluster_label] = model parameters
        self.cluster_labels = {}    # [cid] = cluster_label
        self.aggregated_parameters_global = None
        self.cluster_status = 0 # 0: not started, 1: to cluster, 2: done
        self.n_clusters = 1 # current number of clusters
        self.real_n_clusters = 1
        self.accuracy_trend = [] # accuracy trend for clustering


    # Override configure_fit method to add custom configuration
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)      # Config sent to clients during training 
            if self.cluster_status == 1:
                config["extract_descriptors"] = True

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Clustered training
        if self.cluster_status == 2:
            return [(client,
                     FitIns(self.aggregated_cluster_parameters[self.cluster_labels[client.cid]], \
                            config)) for client in clients]
        
        # Global training if not yet clustering
        else:
            fit_ins = FitIns(parameters, config)
            return [(client, fit_ins) for client in clients]

    
    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        ###############################################################################            
        # Clustering
        ###############################################################################
        if self.cluster_status == 1:
            print(f"\033[91mRound {server_round} - Clustering clients...\033[0m")
            self.cluster_status = 2

            # Extract & scale client descriptors and self-assigned client ids, FLWR cids
            client_descr, client_id_plot, client_cid_list  = [], [], []
            for proxy, res in results:
                if cfg.extended_descriptors:
                    if cfg.selected_descriptors == "Pxy":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using extended Pxy descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["loss_pc_mean"]) + \
                                            json.loads(res.metrics["loss_pc_std"]) + \
                                            json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_std"]))
                    elif cfg.selected_descriptors == "Py":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using extended Py descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["loss_pc_mean"]) + \
                                            json.loads(res.metrics["loss_pc_std"]))
                    elif cfg.selected_descriptors == "Px":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using extended Px descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_std"]))
                    elif cfg.selected_descriptors == "Px_cond":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using extended Px_cond descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_std"]) + \
                                            json.loads(res.metrics["latent_space_cond_mean"]) + \
                                            json.loads(res.metrics["latent_space_cond_std"]))
                    elif cfg.selected_descriptors == "Pxy_cond":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using extended Pxy_cond descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_std"]) + \
                                            json.loads(res.metrics["latent_space_cond_mean"]) + \
                                            json.loads(res.metrics["latent_space_cond_std"]) + \
                                            json.loads(res.metrics["loss_pc_mean"]) + \
                                            json.loads(res.metrics["loss_pc_std"]))
                    elif cfg.selected_descriptors == "Px_label_long":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using extended Px_label_long descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_std"]) + \
                                            json.loads(res.metrics["latent_space_mean_by_label"]) + \
                                            json.loads(res.metrics["latent_space_std_by_label"]))
                    elif cfg.selected_descriptors == "Px_label_short":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using extended Px_label_short descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_std"]) + \
                                            json.loads(res.metrics["latent_space_mean_by_label"]) + \
                                            json.loads(res.metrics["latent_space_std_by_label"]))
                else:    
                    if cfg.selected_descriptors == "Pxy":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using basic Pxy descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["loss_pc_mean"]) + \
                                            json.loads(res.metrics["latent_space_mean"]))
                    elif cfg.selected_descriptors == "Py":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using basic Py descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["loss_pc_std"]))
                    elif cfg.selected_descriptors == "Px":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using basic Px descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]))
                    elif cfg.selected_descriptors == "Px_cond":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using basic Px_cond descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_cond_mean"]))
                    elif cfg.selected_descriptors == "Pxy_cond":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using basic Pxy_cond descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_cond_mean"]) + \
                                            json.loads(res.metrics["loss_pc_mean"]))
                    elif cfg.selected_descriptors == "Px_label_long":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using basic Px_label_long descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_mean_by_label"]))
                    elif cfg.selected_descriptors == "Px_label_short":
                        if res.metrics["cid"] == 1:
                            print(f"\033[91mClustering using basic Px_label_short descriptors\033[0m")
                        client_descr.append(json.loads(res.metrics["latent_space_mean"]) + \
                                            json.loads(res.metrics["latent_space_mean_by_label"]))                        
                        
                client_id_plot.append(res.metrics["cid"])
                client_cid_list.append(proxy.cid)

            
            # scaling
            client_descr = self.descriptors_scaler.scale(np.array(client_descr))
            print(f"\033[91mRound {server_round} - Shape descriptors {client_descr.shape}\033[0m")
            # print(f"\033[91mRound {server_round} - Scaled client descriptors {client_descr}\033[0m")
            
            # Apply PCA to reduce the data to 2D for visualization
            X_reduced = PCA(n_components=2).fit_transform(client_descr)

            # temp save
            # save descriptors and cid list
            # np.save(f'results/client_descr.npy', client_descr)
            # np.save(f'results/client_id_plot.npy', client_id_plot)

            # KMeans
            if cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD == 1:
                range_n_clusters = range(2, cfg.n_clients)
                # Store inertia (sum of squared distances to centroids) and silhouette scores
                inertia, silhouette_scores = [], []
                for n_clusters in range_n_clusters:
                    kmeans = KMeans(n_clusters=n_clusters, random_state=cfg.random_seed)
                    cluster_labels = kmeans.fit_predict(client_descr)
                    inertia.append(kmeans.inertia_)
                    # Calculate silhouette score and append
                    silhouette_avg = silhouette_score(client_descr, cluster_labels)
                    silhouette_scores.append(silhouette_avg)
            
                # Plot inertia (Elbow Method) and silhouette scores
                utils.plot_elbow_and_silhouette(range_n_clusters, inertia, silhouette_scores, server_round)

                # Identify the best number of clusters based on the highest silhouette score
                best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]
                clustering = KMeans(n_clusters=best_n_clusters, random_state=cfg.random_seed)
                cluster_labels = clustering.fit_predict(client_descr)
                # Calculate and save centroids
                _ = utils.calculate_centroids(client_descr, clustering, cluster_labels)
                utils.cluster_plot(X_reduced, cluster_labels, client_id_plot, server_round, name="KMeans")

            # DBSCAN
            elif cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD == 2:
                clustering = DBSCAN(eps=0.5, min_samples=2)  # You can tune the parameters `eps` and `min_samples`
                cluster_labels = clustering.fit_predict(client_descr)
                if min(cluster_labels) < 0: # -1 is for outliers
                    cluster_labels = cluster_labels + abs(min(cluster_labels)) # TODO wrong, outliers are not the same cluster
                # Calculate and save centroids
                _ = utils.calculate_centroids(client_descr, clustering, cluster_labels)
                utils.cluster_plot(X_reduced, cluster_labels, client_id_plot, server_round, name="DBSCAN")
            
            # HDBSCAN
            elif cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD == 3:
                clustering = HDBSCAN(min_cluster_size=2)  # You can tune the parameters `min_cluster_size` and `min_samples`
                # clustering = HDBSCAN(min_cluster_size=5)  # You can tune the parameters `min_cluster_size` and `min_samples`
                cluster_labels = clustering.fit_predict(client_descr) # Note negative values are outliers, here I make them positive for visualization
                if min(cluster_labels) < 0: # -1 is for outliers
                    cluster_labels = cluster_labels + abs(min(cluster_labels))
                # Calculate and save centroids
                _ = utils.calculate_centroids(client_descr, clustering, cluster_labels)
                utils.cluster_plot(X_reduced, cluster_labels, client_id_plot, server_round, name="HDBSCAN")

            # DBSCAN_no_outliers
            elif cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD == 4:
                
                # Calculate eps
                nbrs = NearestNeighbors(n_neighbors=2).fit(client_descr)
                distances, _ = nbrs.kneighbors(client_descr)
                sorted_distances = np.sort(distances[:, 1])
                kneedle = KneeLocator(range(len(sorted_distances)), sorted_distances, curve='convex', direction='increasing')
                # print(f"kneedle.elbow: {sorted_distances[kneedle.elbow]}")
                eps = sorted_distances[kneedle.elbow] * cfg.eps_scaling  

                # Use this final eps for DBSCAN
                clustering = DBSCAN(eps=eps, min_samples=2)
                dbscan_cluster_labels = clustering.fit_predict(client_descr)

                # Find the number of valid clusters (ignoring noise points, -1)
                dbscan_valid_clusters = len(set(dbscan_cluster_labels)) - (1 if -1 in dbscan_cluster_labels else 0)

                # Reassign noise points (-1) to new clusters starting from num_valid_clusters
                cluster_labels = dbscan_cluster_labels.copy()
                next_cluster_label = dbscan_valid_clusters  # Start assigning new cluster numbers

                # Loop over the labels and assign new labels to noise points
                for i, label in enumerate(dbscan_cluster_labels):
                    if label == -1:  # Noise point detected
                        cluster_labels[i] = next_cluster_label
                        next_cluster_label += 1  # Move to the next cluster number

                # Output the final eps, the number of clusters, and the new labels
                print(f"Number of clusters (including reassigned noise points): {len(set(cluster_labels))}")
                print(f"Cluster labels after reassigning noise points: {cluster_labels}")
                self.real_n_clusters = np.load(f'../data/cur_datasets/n_clusters.npy').item()

                _ = utils.calculate_centroids(client_descr, clustering, cluster_labels)
                utils.cluster_plot(X_reduced, cluster_labels, client_id_plot, server_round, name="DBSCAN")

            # DBSCAN_no_outliers
            elif cfg.cfl_oneshot_CLIENT_CLUSTER_METHOD == 5:
                # known prior number of clusters
                n_clusters = np.load(f'../data/cur_datasets/n_clusters.npy').item()
                self.real_n_clusters = n_clusters
                
                # Use Kmeans for supervised clustering
                clustering = KMeans(n_clusters=n_clusters, random_state=cfg.random_seed)
                cluster_labels = clustering.fit_predict(client_descr)
                # Calculate and save centroids
                _ = utils.calculate_centroids(client_descr, clustering, cluster_labels)
                utils.cluster_plot(X_reduced, cluster_labels, client_id_plot, server_round, name="KMeans")
            
            else:
                print("Invalid clustering method!")

            # Update results and assign clusters
            self.n_clusters = max(cluster_labels) + 1
            self.cluster_labels = {cid: cluster_labels[i] for i, cid in enumerate(client_cid_list)}
            # Save cluster labels
            cluster_labels_inference = {cid: cluster_labels[i] for i, cid in enumerate(client_id_plot)}
            np.save(f'results/{self.path}/cluster_labels_inference_{cfg.non_iid_type}_n_clients_{cfg.n_clients}.npy', cluster_labels_inference)
            
            # print(f"\033[91mRound {server_round} - Identified {self.n_clusters} - clusters ({self.cluster_labels.values()} - client cid {client_id_plot})\033[0m")

            # Assuming cluster_labels is a dictionary where keys are client IDs and values are cluster labels
            cluster_labels_list = list(self.cluster_labels.values())
            client_id_plot_sorted, cluster_labels_sorted = zip(*sorted(zip(client_id_plot, cluster_labels_list)))

            print(f"\033[91mRound {server_round} - Identified {self.n_clusters} clusters (Cluster labels: {cluster_labels_sorted} - Client IDs: {client_id_plot_sorted})\033[0m")

        ################################################################################
        # Federated averaging aggregation
        ################################################################################
        # Federated averaging - from traditional code
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        cur_round_cids = [proxy.cid for proxy, _ in results]
        
        # Clustered, update to cluster models
        if self.cluster_status == 2:
            # Split aggregation into clusters
            client_clusters = {i: [] for i in range(self.n_clusters)}
            for i in range(cfg.n_clients):
                cur_cid = cur_round_cids[i]
                cur_cluster = self.cluster_labels[cur_cid]
                client_clusters[cur_cluster].append(weights_results[i])

            # Aggregate each cluster
            for cl, param in client_clusters.items():
                self.aggregated_cluster_parameters[cl] = ndarrays_to_parameters(aggregate(param))
        # Not yet clustered, update to global model
        else:
            self.aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))
        
        # Aggregate custom metrics if aggregation fn was provided   NO FIT METRICS AGGREGATION FN PROVIDED - SKIPPED FOR NOW
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
        ################################################################################
        # Save model
        ################################################################################
        # Clustered, save cluster models
        if self.cluster_status == 2:
            # Save the aggregated cluster models
            for cl, params in self.aggregated_cluster_parameters.items():
                print(f"Saving round {server_round} aggregated_cluster_parameters_{cl}...")
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(params)
                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                # Overwrite the model
                torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{cl}.pth")
        # Not yet clustered, save global model
        else:
            print(f"Saving round {server_round} aggregated_parameters...")
            aggregated_ndarrays: List[np.ndarray] = parameters_to_ndarrays(self.aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Overwrite the model 
            torch.save(self.model.state_dict(), f"checkpoints/{self.path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_server.pth")
        
        return self.aggregated_parameters_global, aggregated_metrics
   
   
    # Override configure_evaluate method to add custom configuration
    def configure_evaluate(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)      # Config sent to clients during evaluation
            
        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        # Clustered eval
        if self.cluster_status == 2:
            return [(client,
                     EvaluateIns(self.aggregated_cluster_parameters[self.cluster_labels[client.cid]], \
                            config)) for client in clients]
                
        # Global eval if not yet clustering
        else:
            evaluate_ins = EvaluateIns(parameters, config)
            return [(client, evaluate_ins) for client in clients]

 
    # Override
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        # Clustering requirements detection
        print(f"\033[93mRound {server_round} - Aggregated loss: {loss_aggregated} - Aggregated metrics: {metrics_aggregated}\033[0m")
        self.accuracy_trend.append(metrics_aggregated["accuracy"])
        
        if self.cluster_status == 0: 
            # Update the max_latent_space for the next round
            max_client_latent_space = max([res.metrics["max_latent_space"] for _, res in results])
            global MAX_LATENT_SPACE 
            MAX_LATENT_SPACE = 1.02 * max_client_latent_space 
            
        # Cluster requirements check
        if self.cluster_status == 0:
            if server_round >= 3:
                d_accuracy = np.diff(self.accuracy_trend)
                if any(d_accuracy < cfg.th_round):
                    print(f"\033[93mRound {server_round} - Threshold reached \033[0m")
                    self.cluster_status = 1
                elif server_round >= 0.8 * cfg.n_rounds:
                    print(f"\033[93mRound {server_round} - Threshold not reached, but 80% of rounds done\033[0m")
                    self.cluster_status = 1
            # elif server_round > 3:
            #     self.cluster_status = 1
            else:
                print(f"\033[93mRound {server_round} - No need to cluster yet\033[0m")
            
            print(f"\033[93mRound {server_round} - Will be clustering next round\033[0m") \
                if self.cluster_status == 1 else None

        return loss_aggregated, metrics_aggregated
















# Main
def main() -> None:
    # Get arguments
    parser = argparse.ArgumentParser(description='Clustered Federated Learning - Server')
    parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
    args = parser.parse_args()

    utils.set_seed(cfg.random_seed + args.fold)
    start_time = time.time()
    exp_path = utils.create_folders()
    device = utils.check_gpu()
    in_channels = utils.get_in_channels()
    model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)
    descriptors_scaler = client_descr_scaling(scaling_method=cfg.cfl_oneshot_CLIENT_SCALING_METHOD,
                                              scaler=MinMaxScaler(),
                                              )
    
    # Define strategy
    strategy = SaveModelStrategy(
        # self defined
        model=model,
        path=exp_path,
        descriptors_scaler=descriptors_scaler,
        # super
        min_fit_clients=cfg.n_clients, # always all training
        min_evaluate_clients=cfg.n_clients, # always all evaluating
        min_available_clients=cfg.n_clients, # always all available
        evaluate_metrics_aggregation_fn=weighted_average,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=fit_config,
    )

    # Start Flower server for three rounds of federated learning
    history = fl.server.start_server(
        server_address=f"{cfg.ip}:{cfg.port}",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.n_rounds),
        strategy=strategy,
    )

    # Convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/{exp_path}/distributed_metrics_{args.fold}.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    # Plot client training loss and accuracy
    utils.plot_all_clients_metrics(fold=args.fold)

    # Plots and Evaluation the model on the client datasets
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, show=False, fold=args.fold)
    
    # Read cluster centroids from json - for test-time inference
    cluster_centroids = np.load(f'results/{exp_path}/centroids_{cfg.non_iid_type}_n_clients_{cfg.n_clients}.npy', allow_pickle=True).item()
    if cfg.selected_descriptors == "Pxy":
        cluster_centroids = {label: centroid[cfg.n_metrics_descriptors*cfg.len_metric_descriptor:] for label, centroid in cluster_centroids.items()} # only latent space
        print(f"\033[93mCluster centroids: {cluster_centroids}\033[0m\n")
    elif cfg.selected_descriptors == "Px":
        print(f"\033[93mCluster centroids: {cluster_centroids}\033[0m\n") # only latent space
    elif cfg.selected_descriptors == "Py":
        # print in red color
        print(f"\033[93mYou cannot use Py at inference, dummy guy! I will read cluster assignement during training for inference\033[0m\n")
        cluster_labels_inference = np.load(f'results/{exp_path}/cluster_labels_inference_{cfg.non_iid_type}_n_clients_{cfg.n_clients}.npy', allow_pickle=True).item()
        print(f"\033[93mCluster labels: {cluster_labels_inference}\033[0m\n")
    elif cfg.selected_descriptors in ["Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short"]:
        cluster_centroids = {label: centroid[:cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor] for label, centroid in cluster_centroids.items()}
        print(f"\033[93mCluster centroids: {cluster_centroids}\033[0m\n") # only latent space
    else:
        raise ValueError("Invalid selected_descriptors")
    # Read cluster assignement during training for inference (known)
    cluster_labels_inference = np.load(f'results/{exp_path}/cluster_labels_inference_{cfg.non_iid_type}_n_clients_{cfg.n_clients}.npy', allow_pickle=True).item()
    print(f"\033[93mRead cluster assignement during training for inference\033[0m\n")
    print(f"\033[93mCluster labels: {cluster_labels_inference}\033[0m\n")
    
    # Load global model for evaluation
    evaluation_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                          input_size=cfg.input_size).to(device)
    evaluation_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_server.pth", weights_only=False))

    # Evaluate the model on the client datasets    
    losses, accuracies = [], []
    losses_known, accuracies_known = [], []
    for client_id in range(cfg.n_clients):
        test_x, test_y = [], []
        if not cfg.training_drifting:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}.npy', allow_pickle=True).item()
            test_x = cur_data['test_features'] if in_channels == 3 else cur_data['test_features'].unsqueeze(1)
            test_y = cur_data['test_labels']
        else:
            cur_data = np.load(f'../data/cur_datasets/client_{client_id}_round_-1.npy', allow_pickle=True).item()
            test_x = cur_data['features'] if in_channels == 3 else cur_data['features'].unsqueeze(1)
            test_y = cur_data['labels']
        
        # Create test dataset and loader
        test_dataset = models.CombinedDataset(test_x, test_y, transform=None)
        test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)
    
        # --- Test-time inference: check closest cluster ---
        # Extract descriptors, scaling
        descriptors = models.ModelEvaluator(test_loader=test_loader, device=device).extract_descriptors_inference(
                                                    model=evaluation_model, max_latent_space=MAX_LATENT_SPACE)
        
        if cfg.selected_descriptors == "Pxy":
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))
            descriptors = descriptors[:, cfg.n_metrics_descriptors*cfg.len_metric_descriptor:] # only latent space
        elif cfg.selected_descriptors == "Px":
            descriptors = descriptors[cfg.n_metrics_descriptors*cfg.len_metric_descriptor:] # only latent space 
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))
        elif cfg.selected_descriptors in ["Px_cond", "Pxy_cond", "Px_label_long", "Px_label_short"]:
            descriptors = descriptors_scaler.scale(descriptors.reshape(1,-1))
            descriptors = descriptors[:, :cfg.n_latent_space_descriptors*cfg.len_latent_space_descriptor] # only latent space
        else:
            raise ValueError("Invalid selected_descriptors")
    
        # Find the closest cluster centroid
        client_cluster = min(cluster_centroids, key=lambda k: np.linalg.norm(descriptors - cluster_centroids[k]))
        
        # Load respective cluster model
        cluster_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                        input_size=cfg.input_size).to(device)
        cluster_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{client_cluster}.pth", weights_only=False))
        
        # Evaluate
        loss_test, accuracy_test = models.simple_test(cluster_model, device, test_loader)
        print(f"\033[93mClient {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f} - Closest centroid {client_cluster}\033[0m")
        accuracies.append(accuracy_test)
        losses.append(loss_test)
        
        
        # --- Participating clients: assign known cluster ---
        client_cluster = cluster_labels_inference[client_id]

        # Load respective cluster model
        cluster_model = models.models[cfg.model_name](in_channels=in_channels, num_classes=cfg.n_classes, \
                                        input_size=cfg.input_size).to(device)
        cluster_model.load_state_dict(torch.load(f"checkpoints/{exp_path}/{cfg.non_iid_type}_n_clients_{cfg.n_clients}_cluster_{client_cluster}.pth", weights_only=False))
        
        # Evaluate
        loss_test, accuracy_test = models.simple_test(cluster_model, device, test_loader)
        print(f"\033[93mClient (known) {client_id} - Test Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f} - Closest centroid {client_cluster}\033[0m")
        accuracies_known.append(accuracy_test)
        losses_known.append(loss_test)



    # print average loss and accuracy
    print(f"\n\033[93mAverage Loss: {np.mean(losses):.3f}, Average Accuracy: {np.mean(accuracies)*100:.2f}\033[0m")
    print(f"\033[93mAverage Loss (known): {np.mean(losses_known):.3f}, Average Accuracy (known): {np.mean(accuracies_known)*100:.2f}\033[0m")
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    
    # Save metrics as numpy array
    metrics = {
        "loss": losses,
        "accuracy": accuracies,
        "average_loss": np.mean(losses),
        "average_accuracy": np.mean(accuracies),
        "loss_known": losses_known,
        "accuracy_known": accuracies_known,
        "average_loss_known": np.mean(losses_known),
        "average_accuracy_known": np.mean(accuracies_known),
        "time": round((time.time() - start_time)/60, 2),
        "identified_clusters": strategy.n_clusters,
        "real_clusters": strategy.real_n_clusters,
    }
    np.save(f'results/{exp_path}/test_metrics_fold_{args.fold}.npy', metrics)
    
    time.sleep(1)
    
if __name__ == "__main__":
    main()
