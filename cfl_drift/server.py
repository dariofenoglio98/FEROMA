"""
This code implements the FedAvg, when it starts, the server waits for the clients to connect. When the established number 
of clients is reached, the learning process starts. The server sends the model to the clients, and the clients train the 
model locally. After training, the clients send the updated model back to the server. Then client models are aggregated 
with FedAvg. The aggregated model is then sent to the clients for the next round of training. The server saves the model 
and metrics after each round.

This is code is set to be used locally, but it can be used in a distributed environment by changing the server_address.
In a distributed environment, the server_address should be the IP address of the server, and each client machine should 
run the appopriate client code (client.py).

METHOD: 

"""


## NOTE: p(x) can be otbained accurately with a generative model, and it does not requires samples on the server side
## NOTE: p(x) describe a the distribution of the data, and with the latent space extraction we are trying to describe 
##       the distribution of the data in a lower dimension but so far we are using only the mean which is not enough. 
##       Think about adding also the variance to the latent space and others

## **What if**: we train locally with DP a VAE to learn p(x) in the fist round or when needed (we just need one round 
##       and something like 30 epochs), then use the VAE to define the distribution of the data (or generate the data
##       on the server and then use others method to perform cluster on client data - ask daniel for Gaussian Mixture).
##       This can still be combined with the label-wise loss for evaluating the p(y) and p(x|y) on the server side.


# Libraries
import flwr as fl
from modified_flwr import app
import numpy as np
from typing import List, Tuple, Union, Optional, Dict
from flwr.common import Parameters, Scalar, Metrics
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common import FitRes
import argparse
import torch
from torch.utils.data import DataLoader
import os
from logging import WARNING
from flwr.common.logger import log
from collections import OrderedDict
import json
import time
import pandas as pd
import public.config as cfg
import public.utils as utils
import public.models as models
from sklearn.model_selection import train_test_split
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitRes,
    FitIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common import NDArray, NDArrays
from functools import reduce
from flwr.server.client_manager import ClientManager
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from modified_flwr.server import Server


# Define the max latent space as global variable
max_latent_space = 2


# Config_client
def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    config = {
        "current_round": server_round,
        "local_epochs": cfg.local_epochs,
        "tot_rounds": cfg.n_rounds,
        "min_latent_space": 0,
        "max_latent_space": max_latent_space,
        "extract_descriptors": False,
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

def aggregate_fit(
    self,
    server_round: int,
    results: List[Tuple[ClientProxy, FitRes]],
    failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
    """Aggregate fit results using weighted average."""
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
    parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

    # Aggregate custom metrics if aggregation fn was provided
    metrics_aggregated = {}
    if self.fit_metrics_aggregation_fn:
        fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
        metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
    elif server_round == 1:  # Only log this warning once
        log(WARNING, "No fit_metrics_aggregation_fn provided")

    return parameters_aggregated, metrics_aggregated

# Custom strategy to save model after each round
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.dataset = dataset
        self.client_cid_list = []
        self.aggregated_cluster_parameters = []
        self.cluster_labels = {}
        self.aggregated_parameters_global = None
        self.first_cluster_done = False
        self.cluster_do = False
        self.accuracy_reached = 0
        self.no_clusters_round_counter = 0

    # Override aggregate_fit method to add saving functionality
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
        descriptor_extraction: bool = False,
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model weights using weighted average and store checkpoint"""
        
        
        # Federated averaging - from traditional code
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate custom metrics if aggregation fn was provided   NO FIT METRICS AGGREGATION FN PROVIDED - SKIPPED FOR NOW
        aggregated_metrics = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            aggregated_metrics = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")

        # VARIABLE TO START DESCRIPTOR EXTRACTION
        if self.no_clusters_round_counter > 0:
            self.no_clusters_round_counter -= 1


        ################################################################################
        # Extract descriptors from results - clustering - aggregation
        ################################################################################
        if self.accuracy_reached > cfg.th_accuracy and descriptor_extraction and self.no_clusters_round_counter==0:
            print(f"\033[93mCLUSTERING...\033[0m")
            self.cluster_do = True
            self.no_clusters_round_counter = 5
            # NEW CLUSTERING ROUND - RECEIVE DESCRIPTORS FROM CLIENTS - CLUSTER - MODEL AGGREGATION
            # Extract descriptors from results
            client_descr_scaled = self.extract_descriptors_from_results(results)
            # Cluster clients
            centroids = self.clustering(server_round, client_descr_scaled)
            # Aggregate model TODO: assign the closest cluster model to each clients
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            
            ### ------- not finished yet, think about it before ------- ###
            if not self.first_cluster_done:
                self.first_cluster_done = True
                self.centroids = centroids # save centroids for the next clustering round
                # Split aggregation into clusters
                client_clusters = {i: [] for i in range(self.n_clusters)}
                for i, cluster in enumerate(self.cluster_labels.values()):
                    client_clusters[cluster].append(weights_results[i]) 
                # Aggregate each cluster
                self.aggregated_cluster_parameters = []
                for cluster in client_clusters.values():
                        self.aggregated_cluster_parameters.append(ndarrays_to_parameters(aggregate(cluster))) # problem you are clustering the global model.. always the same
            else:
                # Assign the closest previous cluster model to the new cluster
                for i, centroid in centroids.items():
                    distances = [np.linalg.norm(centroid - c) for c in self.centroids.values()]
                    closest_cluster = np.argmin(distances)
                    self.cluster_labels[i] = closest_cluster
                    self.aggregated_cluster_parameters[closest_cluster] = ndarrays_to_parameters(aggregate([self.aggregated_cluster_parameters[closest_cluster], weights_results[i]]))
            ### ------------------------------------------------------- ###   
                  
            # # Update the max_latent_space for the next round
            max_client_latent_space = max([res.metrics["max_latent_space"] for _, res in results])
            global max_latent_space 
            max_latent_space = 1.2 * max_client_latent_space 

            # Saving evaluation model
            print(f"Saving round {server_round} aggregated_parameters...")
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(self.aggregated_parameters_global)
            # Convert `List[np.ndarray]` to PyTorch`state_dict`
            params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)
            # Save the model
            torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_evaluation_clusters.pth")
            
            return self.aggregated_parameters_global, aggregated_metrics
        
        
        ################################################################################
        # Federated averaging aggregation and cluster aggregation
        ################################################################################
        elif not descriptor_extraction:
            # Convert results
            weights_results = [
                (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
                for _, fit_res in results
            ]
            
            # i can skip this part when cluster_done is False, but i still need aggregated_parameters_global to pass
            aggregated_parameters_global = ndarrays_to_parameters(aggregate(weights_results))   # Global aggregation - traditional - no clustering
            self.aggregated_parameters_global = aggregated_parameters_global

            if self.cluster_do:
                print(f"\033[93mAGGREGATING CLUSTERS...\033[0m")
                # Split aggregation into clusters
                client_clusters = {i: [] for i in range(self.n_clusters)}
                for i, cluster in enumerate(self.cluster_labels.values()):
                    client_clusters[cluster].append(weights_results[i])
                    
                # Aggregate each cluster
                self.aggregated_cluster_parameters = []
                for cluster in client_clusters.values():
                        self.aggregated_cluster_parameters.append(ndarrays_to_parameters(aggregate(cluster)))
        
            # Save model
            if aggregated_parameters_global is not None:

                print(f"Saving round {server_round} aggregated_parameters...")
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters_global)
                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                # Save the model
                torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{server_round}.pth")
            
            # Save the aggregated cluster models
            for i, aggregated_cluster_parameters in enumerate(self.aggregated_cluster_parameters):
                print(f"Saving round {server_round} aggregated_cluster_parameters_{i}...")
                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_cluster_parameters)
                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(self.model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                self.model.load_state_dict(state_dict, strict=True)
                # Save the model
                torch.save(self.model.state_dict(), f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{server_round}_cluster_{i}.pth")
            
            return aggregated_parameters_global, aggregated_metrics
        
        
        ############################################################################
        # No aggregation 
        ############################################################################
        else:
            return None, {}
    
    
    ############################################################################################################
    # Aggregate evaluation results
    ############################################################################################################
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
        descriptor_extraction: bool = False,
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
        
        print(f"\033[93mRound {server_round} - Aggregated loss: {loss_aggregated} - Aggregated metrics: {metrics_aggregated}\033[0m")
        
        if metrics_aggregated["accuracy"] > cfg.th_accuracy:
            self.accuracy_reached = metrics_aggregated["accuracy"]

        return loss_aggregated, metrics_aggregated
    
    
    ############################################################################################################
    # Extract descriptors from results
    ############################################################################################################
    def extract_descriptors_from_results(self, results: List[Tuple[ClientProxy, EvaluateRes]]):
            # Update client cid list   
            self.client_cid_list = []
            for client in results:
                self.client_cid_list.append(client[0].cid)     # Automatically assigned cid by Flower
            
            # Extract client descriptors
            client_descr = []
            self.client_cid = []
            for _, res in results:
                # res.num_examples, res.metrics
                loss_pc = json.loads(res.metrics["loss_pc"])
                latent_space = json.loads(res.metrics["latent_space"])
                self.client_cid.append(res.metrics["cid"])
                # concatenate the descriptors
                client_descr.append(loss_pc + latent_space)
            
            # Normalizing descriptor matrix
            client_descr = np.array(client_descr)
            # 1. Normalize each column between 0 and 1 #TODO: test both 1 and 2 methods
            scaler = MinMaxScaler() # StandardScaler()
            client_descr_scaled_1 = scaler.fit_transform(client_descr)
            # print(f"scaled_data: {client_descr_scaled_1}")
            # 2. Normalize by group of descriptors #TODO: test both 1 and 2 methods
            loss_pc = client_descr[:, :cfg.n_classes]
            latent_space = client_descr[:, cfg.n_classes:]
            scaled_loss_pc = scaler.fit_transform(loss_pc.reshape(-1, 1)).reshape(loss_pc.shape)  
            latent_space_pc = scaler.fit_transform(latent_space.reshape(-1, 1)).reshape(latent_space.shape)
            client_descr_scaled_2 = np.hstack((scaled_loss_pc, latent_space_pc)) # TODO: we can also weight them (loss and latent) differently here, by multiplying them by a factor
            # print(f"scaled_data_2: {client_descr_scaled_2}")
            
            print(f"client cid: {self.client_cid}")
            print(f"client cid list: {self.client_cid_list}")
            
            # TODO: choose the best method for normalization
            return client_descr_scaled_2


    ############################################################################################################
    # Cluster clients
    ############################################################################################################
    def clustering(self, server_round: int, client_descr_scaled):
        print(f"\033[91mRound {server_round} - Clustering clients...\033[0m")
        
        # Range of cluster numbers to try
        range_n_clusters = range(2, client_descr_scaled.shape[0])  # Adjust based on your data size

        # Store inertia (sum of squared distances to centroids) and silhouette scores
        inertia = []
        silhouette_scores = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(client_descr_scaled)
            inertia.append(kmeans.inertia_)
            
            # Calculate silhouette score and append
            silhouette_avg = silhouette_score(client_descr_scaled, cluster_labels)
            silhouette_scores.append(silhouette_avg)
    
        # # Plot inertia (Elbow Method) and silhouette scores
        utils.plot_elbow_and_silhouette(range_n_clusters, inertia, silhouette_scores, server_round)

        # Identify the best number of clusters based on the highest silhouette score
        best_n_clusters = range_n_clusters[np.argmax(silhouette_scores)]

        # Apply PCA to reduce the data to 2D for visualization
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(client_descr_scaled)
        # Apply KMeans with the best number of clusters
        kmeans_best = KMeans(n_clusters=best_n_clusters, random_state=42)
        cluster_labels_kmeans = kmeans_best.fit_predict(client_descr_scaled)
        centroids_kmeans = kmeans_best.cluster_centers_
        centroids_kmeans_reduced = pca.transform(centroids_kmeans)
        X_reduced_kmeans_centroids = np.vstack((X_reduced, centroids_kmeans_reduced))
        cluster_labels_kmeans_centroids = np.append(cluster_labels_kmeans, [f"C{n}" for n in np.unique(cluster_labels_kmeans)])
        client_cid_ext = self.client_cid + [f"C{n}" for n in np.unique(cluster_labels_kmeans)]
        # Plot the identified clusters with the best score/number of clusters
        utils.cluster_plot(X_reduced_kmeans_centroids, cluster_labels_kmeans_centroids, client_cid_ext, server_round, name="KMeans")

        # Apply DBSCAN (doesn't require specifying the number of clusters)
        # clustering = DBSCAN(eps=0.5, min_samples=2)  # You can tune the parameters `eps` and `min_samples`
        # cluster_labels_dbscan = clustering.fit_predict(client_descr_scaled)
        # print(f"DBSCAN cluster_labels: {cluster_labels_dbscan}")
        # Plot the identified DBSCAN clusters
        # utils.cluster_plot(X_reduced, cluster_labels_dbscan, client_cid, server_round, name="DBSCAN")
        
        # HDBSCAN
        clustering = HDBSCAN(min_cluster_size=2, store_centers='centroid')  # You can tune the parameters `min_cluster_size` and `min_samples`
        cluster_labels_hdbscan = clustering.fit_predict(client_descr_scaled) # Note negative values are outliers, here I make them positive for visualization
        centroids_hdbscan, cluster_labels_hdbscan = self.calculate_hdbscan_centroids(client_descr_scaled, clustering, cluster_labels_hdbscan)
        # Plot the identified HDBSCAN clusters
        centroids_hdbscan_reduced = pca.transform(list(centroids_hdbscan.values()))
        X_reduced_centroids = np.vstack((X_reduced, centroids_hdbscan_reduced))
        cluster_labels_hdbscan_centroids = np.append(cluster_labels_hdbscan, [f"C{n}" for n in centroids_hdbscan.keys()])
        client_cid_ext = self.client_cid + [f"C{n}" for n in centroids_hdbscan.keys()]
        utils.cluster_plot(X_reduced_centroids, cluster_labels_hdbscan_centroids, client_cid_ext, server_round, name="HDBSCAN")
        
        # Choose the best clustering methods
        self.n_clusters = max(cluster_labels_hdbscan) + 1  # Best number of clusters
        self.cluster_labels = {cid: cluster_labels_hdbscan[i] for i, cid in enumerate(self.client_cid_list)}  # Best clustering method
        centroids = {n: centroids_hdbscan[n] for n in range(self.n_clusters)}
        print(f"\033[91mRound {server_round} - Identified {self.n_clusters} clusters ({self.cluster_labels.values()})\033[0m")
        
        return centroids


    # def calculate_hdbscan_centroids(self, data, clustering, cluster_labels_hdbscan):
    #     """
    #     Calculate centroids for all clusters, including outliers, using HDBSCAN.

    #     Parameters:
    #     - data: The dataset used for clustering (numpy array or pandas DataFrame).
    #     - clustering: The HDBSCAN clustering object after fitting.
    #     - cluster_labels_hdbscan: The cluster labels assigned by HDBSCAN, including outliers.

    #     Returns:
    #     - centroids_dict: A dictionary with cluster labels as keys and centroids as values.
    #     """
    #     # Access non-outlier centroids
    #     centroids_hdbscan = clustering.centroids_

    #     # Create a dictionary to store centroids for all clusters (including outliers)
    #     centroids_dict = {}

    #     # Handle non-outlier centroids
    #     unique_labels = np.unique(cluster_labels_hdbscan)
    #     non_outlier_labels = unique_labels[unique_labels >= 0]  # Ignore outliers labeled as -1

    #     # Add centroids for non-outlier clusters to the dictionary
    #     for label, centroid in zip(non_outlier_labels, centroids_hdbscan):
    #         centroids_dict[label] = centroid

    #     # Handle outliers (clusters labeled as -1, -2, etc.)
    #     outlier_labels = unique_labels[unique_labels < 0]

    #     for outlier_label in outlier_labels:
    #         outlier_points = data[cluster_labels_hdbscan == outlier_label]
    #         if len(outlier_points) > 0:
    #             # Compute the centroid of the outlier points
    #             outlier_centroid = np.mean(outlier_points, axis=0)
    #             centroids_dict[outlier_label] = outlier_centroid

    #     return centroids_dict

    def calculate_hdbscan_centroids(self, data, clustering, cluster_labels_hdbscan):
        """
        Calculate centroids for all clusters, including outliers, using HDBSCAN,
        and shift the labels to be positive for visualization purposes.

        Parameters:
        - data: The dataset used for clustering (numpy array or pandas DataFrame).
        - clustering: The HDBSCAN clustering object after fitting.
        - cluster_labels_hdbscan: The cluster labels assigned by HDBSCAN, including outliers.

        Returns:
        - centroids_dict: A dictionary with adjusted (positive) cluster labels as keys and centroids as values.
        - shifted_labels: The adjusted positive cluster labels for the dataset.
        """
        # Access non-outlier centroids
        centroids_hdbscan = clustering.centroids_

        # Create a dictionary to store centroids for all clusters (including outliers)
        centroids_dict = {}

        # Find the minimum cluster label (for outliers it will be negative)
        min_label = min(cluster_labels_hdbscan)
        shift_value = abs(min_label) if min_label < 0 else 0

        # Shift all cluster labels to make them positive
        shifted_labels = cluster_labels_hdbscan + shift_value

        # Handle non-outlier centroids
        unique_labels = np.unique(cluster_labels_hdbscan)
        non_outlier_labels = unique_labels[unique_labels >= 0]  # Ignore outliers labeled as negative

        # Add centroids for non-outlier clusters to the dictionary with shifted keys
        for label, centroid in zip(non_outlier_labels, centroids_hdbscan):
            centroids_dict[label + shift_value] = centroid

        # Handle outliers (clusters labeled as -1, -2, etc.) and add them to the dictionary
        outlier_labels = unique_labels[unique_labels < 0]

        for outlier_label in outlier_labels:
            outlier_points = data[cluster_labels_hdbscan == outlier_label]
            if len(outlier_points) > 0:
                # Compute the centroid of the outlier points
                outlier_centroid = np.mean(outlier_points, axis=0)
                centroids_dict[outlier_label + shift_value] = outlier_centroid

        return centroids_dict, shifted_labels


    ############################################################################################################
    # Configure fit - Send custom configuration to clients for training
    ############################################################################################################
    # Override configure_fit method to add custom configuration
    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager, descriptor_extraction: bool = False
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)      # Config sent to clients during training 
            
        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        
        if descriptor_extraction:
            print(f"\033[93mDESCRIPTOR EXTRACTION-SENDING BACK GLOBAL MODEL...\033[0m")
            # GLOBAL EVALUATION - SEND GLOBAL MODEL TO ALL CLIENTS FOR DESCRIPTOR EXTRACTION
            config["extract_descriptors"] = True
            evaluate_ins = EvaluateIns(parameters, config)
            return [(client, evaluate_ins) for client in clients]
        else: 
            config["extract_descriptors"] = False
            # if cluster is not done use global otherwise use cluster aggregation
            if not self.cluster_do:
                print(f"\033[93mSENDING BACK GLOBAL MODEL FOR TRAINING...\033[0m")
                # GLOBAL TRAINING
                fit_ins = FitIns(parameters, config)
                return [(client, fit_ins) for client in clients]
            else:           
                # CLUSTERED AGGREGATION - Instrucions for clients - custom fit_ins for each client 
                print(f"\033[93mSENDING BACK CLUSTER MODELS FOR TRAINING...\033[0m")
                client_cids_sampled = [client.cid for client in clients]
                fit_ins = []
                for c in client_cids_sampled:
                    fit_ins.append(FitIns(self.aggregated_cluster_parameters[self.cluster_labels[c]], config))
                return [(client, fit_ins[i]) for i, client in enumerate(clients)]
        
      
    ############################################################################################################
    # Configure evaluate - Send custom configuration to clients for evaluation
    ############################################################################################################  
    # Override configure_evaluate method to add custom configuration
    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager, descriptor_extraction: bool = False
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
        
        if not self.cluster_do:
            print(f"\033[93mSENDING BACK GLOBAL MODEL FOR EVALUATION...\033[0m")
            # GLOBAL EVALUATION - SEND GLOBAL MODEL TO ALL CLIENTS FOR DESCRIPTOR EXTRACTION
            config["extract_descriptors"] = True
            evaluate_ins = EvaluateIns(parameters, config)
            return [(client, evaluate_ins) for client in clients]
        else:
            print(f"\033[93mSENDING BACK CLUSTER MODELS FOR EVALUATION...\033[0m")
            # CLUSTERED AGGREGATION - SEND BACK FOR EVALUATION EACH CLUSTER MODEL TO THE RESPECTIVE CLIENT 
            config["extract_descriptors"] = False
            client_cids_sampled = [client.cid for client in clients]
            evaluate_ins = []
            for c in client_cids_sampled:
                evaluate_ins.append(EvaluateIns(self.aggregated_cluster_parameters[self.cluster_labels[c]], config))
            return [(client, evaluate_ins[i]) for i, client in enumerate(clients)]
        
        
        




# Main
def main() -> None:
    # Get arguments
    parser = argparse.ArgumentParser(description='Dynamic Clustered Federated Learning - Server')
    parser.add_argument('--fold', type=int, default=0, help='Fold number of the cross-validation')
    args = parser.parse_args()
    
    # Start time
    start_time = time.time()

    # Create directories 
    utils.create_folders()

    # Create all data for clients
    utils.generate_dataset()

    # Pick the indipendent test set from each client
    test_x, test_y = [], []
    for client_id in range(cfg.client_number):
        data = np.load(f'./data/client_{client_id}.npy', allow_pickle=True).item()
        test_x.append(data['test_features'])
        test_y.append(data['test_labels'])
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)
    # Split the data into test subsets (final evaluation & server validation)
    test_x_server, test_x, test_y_server, test_y = train_test_split(test_x, test_y, test_size=0.5, random_state=42)

    # Create the datasets
    test_dataset = models.CombinedDataset(test_x, test_y, transform=None)

    # Create the data loaders
    test_loader = DataLoader(test_dataset, batch_size=cfg.test_batch_size, shuffle=False)

    # model and history folder
    device = utils.check_gpu(manual_seed=True, print_info=True)
    model = models.models[cfg.model_name](in_channels=3, num_classes=cfg.n_classes, input_size=cfg.input_size).to(device)

    # Define strategy
    strategy = SaveModelStrategy(
        model=model, # model to be trained
        min_fit_clients=cfg.client_number, #+cfg.n_attackers, # Never sample less than 10 clients for training
        min_evaluate_clients=cfg.client_number, #+cfg.n_attackers,  # Never sample less than 5 clients for evaluation
        min_available_clients=cfg.client_number, #+cfg.n_attackers, # Wait until all 10 clients are available
        fraction_fit=1.0, # Sample 100 % of available clients for training
        fraction_evaluate=1.0, # Sample 100 % of available clients for evaluation
        evaluate_metrics_aggregation_fn=weighted_average,
        fit_metrics_aggregation_fn=weighted_average,
        on_evaluate_config_fn=fit_config,
        on_fit_config_fn=fit_config,
        dataset=cfg.dataset_name,
    )
    
    # server = Server(strategy=strategy)

    print(f"\n\033[94mTraining {cfg.model_name} on {cfg.dataset_name} with {cfg.client_number} clients\033[0m\n")

    # Start Flower server for three rounds of federated learning
    history = app.start_server(
        server=Server(client_manager=SimpleClientManager(), strategy=strategy),
        server_address=f"{cfg.ip}:{cfg.port}",   # 0.0.0.0 listens to all available interfaces
        config=fl.server.ServerConfig(num_rounds=cfg.n_rounds),
        # strategy=strategy,
    )
    # convert history to list
    loss = [k[1] for k in history.losses_distributed]
    accuracy = [k[1] for k in history.metrics_distributed['accuracy']]

    # Save loss and accuracy to a file
    print(f"Saving metrics to as .json in histories folder...")
    with open(f'histories/{cfg.model_name}/{cfg.dataset_name}/distributed_metrics.json', 'w') as f:
        json.dump({'loss': loss, 'accuracy': accuracy}, f)

    # Single Plot
    best_loss_round, best_acc_round = utils.plot_loss_and_accuracy(loss, accuracy, show=False)

    # Load the best model
    model.load_state_dict(torch.load(f"checkpoints/{cfg.model_name}/{cfg.dataset_name}/model_{best_loss_round}.pth", weights_only=False))

    # Evaluate the model on the test set
    loss_test, accuracy_test = models.simple_test(model, device, test_loader)
    print(f"\n\033[93mTest Loss: {loss_test:.3f}, Test Accuracy: {accuracy_test*100:.2f}\033[0m")
    print(f"\033[93mNOTE: global model is evaluated, not correct!\033[0m\n")

    # Print training time in minutes (grey color)
    print(f"\033[90mTraining time: {round((time.time() - start_time)/60, 2)} minutes\033[0m")
    time.sleep(1)
    
if __name__ == "__main__":
    main()
