#!/bin/bash

k_folds=$(python -c "from public.config import k_folds; print(k_folds)")
model_name=$(python -c "from public.config import model_name; print(model_name)")
dataset_name=$(python -c "from public.config import dataset_name; print(dataset_name)")
strategy=$(python -c "from public.config import strategy; print(strategy)")
drifting_type=$(python -c "from public.config import drifting_type; print(drifting_type)")
non_iid_type=$(python -c "from public.config import non_iid_type; print(non_iid_type)")
n_clients=$(python -c "from public.config import n_clients; print(n_clients)")
n_rounds=$(python -c "from public.config import n_rounds; print(n_rounds)")

echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
    MODEL: $model_name\033[0m\n\033[1;36m \
    Dataset: $dataset_name\033[0m\n\033[1;36m \
    Strategy: $strategy\033[0m\n\033[1;36m \
    Drifting type: $drifting_type\033[0m\n\033[1;36m \
    Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
    Number of clients: $n_clients\033[0m\n\033[1;36m \
    Number of rounds: $n_rounds\033[0m\n \
    \033[1;36mK-Folds: $k_folds\033[0m\n"


# K-Fold evaluation, if k_folds > 1
for fold in $(seq 0 $(($k_folds - 1))); do        
    echo -e "\n\033[1;36mStarting fold $((fold + 1))\033[0m\n"

    # Clean and create datasets
    rm -rf data/cur_datasets/* 
    python public/generate_datasets.py --fold "$fold"

    # exit

    cd "$strategy"

    python server.py --fold "$fold" &
    # python dynamic_cluster_global_server.py &
    sleep 2  # Sleep for 2s to give the server enough time to start

    for i in $(seq 0 $(($n_clients - 1))); do
        echo "Starting client ID $i"
        # python client_dynamic.py --id "$i" &
        python client.py --id "$i" --fold "$fold" &
    done

    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait

    # Clean up
    echo "Fold completed correctly"
    trap - SIGTERM 
    # pkill -u dario -f client.py
    pkill -u dariofenoglio -f client.py
    # pkill -u mohan -f client.py
    # pkill -u mohanli -f client.py
    # pkill -u dario -f server.py
    pkill -u dariofenoglio -f sever.py
    # pkill -u mohan -f server.py
    # pkill -u mohanli -f server.py

    # Change back to the root directory
    cd ..
    sleep 3

done

# K-Fold evaluation, if k_folds > 1
if [ "$k_folds" -gt 1 ]; then

    echo -e "\n\033[1;36mAveraging the results of all folds\033[0m\n"
    # Averaging the results of all folds
    python public/average_results.py
    # Plot confidence interval plots
    python public/plots_across_folds.py --dataset "$dataset_name"
fi


echo -e "\n\033[1;36mExperiment completed successfully\033[0m\n"
# kill
trap - SIGTERM && kill -- -$$
