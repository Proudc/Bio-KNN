dataset_list=(big_uniprot big_uniref)
cluster_list=(1 2 4 8)
for dataset in ${dataset_list[@]}; do
    for cluster in ${cluster_list[@]}; do   
        python ./clustering/cluster.py ${dataset} ${cluster}
        PID0=$!
        wait $PID0
    done
done
