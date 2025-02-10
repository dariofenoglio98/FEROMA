# DFUL: Handling Drifting in Federated Learning

**DFUL** is a Federated Learning (FL) framework designed for addressing data drifting during training and testing phase based on their data distribution. This framework aims to enhance the efficiency and accuracy of FL by grouping clients with similar data characteristics. Additionally, CFL addresses data shift and drifting, considering variations in:

- Feature distribution shift: The marginal distributions \(P(X)\) vary across clients.
- Label distribution shift: The marginal distributions \(P(Y)\) vary across clients.
- Concept shift (Same features, different label): The conditional distributions \(P(Y|X)\) vary across clients.
- Concept shift (Same label, different features): The conditional distributions \(P(X|Y)\) vary across clients.


## Features

- **Dynamic Client Clustering**: Clients are aggregated during training based on the distance of their data distribution, optimizing collaboration and model convergence.
- **Data Shift & Drift Handling**: Adapts to data shifts across clients by considering the distributional changes in both input and output spaces (See ANDA for more information).
- **Scalable FL Framework**: Designed to scale across large numbers of clients while maintaining computation and communication efficiency.

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/dariofenoglio98/dful.git
   cd CFL
   ```
2. Set Conda Environment and necessary Libraries
    ```
    conda env create -f environment.yml
    ```
3. or install only the required Python packages
    ```
    pip install -r requirements.txt
    ```

## Usage - Tutorial

## Run Experiments

## License
This project is licensed under the MIT License – see the LICENSE.md file for details.

## Citation
