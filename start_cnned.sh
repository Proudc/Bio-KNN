network=CNNED
loss=triplet
sampling=distance_sampling3
train_ratio=1
target_size=128

dataset_list=(big_uniref)
seed_list=(666)
for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        train_flag=CNNED_${dataset}_${seed}_${target_size}
        echo Training_${train_flag}
        CUDA_VISIBLE_DEVICES=1 python train.py --train_flag ${train_flag} --network_type ${network} --loss_type ${loss} --sampling_type ${sampling} --random_seed ${seed} --root_read_path ./${dataset} --root_write_path /mnt/data_hdd3/czh/Neuprotein/${dataset} --test_epoch 2000 --epoch_num 1000 --dataset_type ${dataset} --train_ratio ${train_ratio} --target_size ${target_size}

        echo Inference_${train_flag}
        CUDA_VISIBLE_DEVICES=1 python train.py --train_flag ${train_flag} --network_type ${network} --loss_type ${loss} --sampling_type ${sampling} --random_seed ${seed} --root_read_path ./${dataset} --root_write_path /mnt/data_hdd3/czh/Neuprotein/${dataset} --test_epoch 2000 --epoch_num 1000 --dataset_type ${dataset} --train_ratio ${train_ratio} --target_size ${target_size} --mode test
    done
done
