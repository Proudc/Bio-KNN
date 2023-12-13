dataset_list=()
cluster_list=(1 2 4 8)
algorithm_list=(agglomerative)
for dataset in ${dataset_list[@]}; do
    for cluster in ${cluster_list[@]}; do   
        for algo in ${algorithm_list[@]}; do
            python ./clustering/vis.py ${dataset} ${cluster} 1000 ${algo}
            PID0=$!
            wait $PID0
        done
    done
done
