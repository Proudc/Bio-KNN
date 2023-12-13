network=CNNED
loss=triplet
sampling=distance_sampling3
train_ratio=0.2
target_size=128
dataset_list=(big_uniprot)
seed_list=(666 555 444)
for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        area_path=./${dataset}/clustering/agglomerative_1000_100_4_None
        train_flag=MYCNN_Ablation_onehead_${dataset}_${seed}_${target_size}
        
        echo Training_${train_flag}
        CUDA_VISIBLE_DEVICES=0 nohup python train.py --train_flag ${train_flag} --network_type ${network} --loss_type ${loss} --sampling_type ${sampling} --random_seed ${seed} --root_read_path ./${dataset} --root_write_path ../${dataset} --test_epoch 2000 --epoch_num 1000 --dataset_type ${dataset} --train_ratio ${train_ratio} --target_size ${target_size} --area_path ${area_path} > train_log/training_${train_flag}.log &
        PID=$!
        wait $PID

        echo Inference_${train_flag}
        CUDA_VISIBLE_DEVICES=0 nohup python train.py --train_flag ${train_flag} --network_type ${network} --loss_type ${loss} --sampling_type ${sampling} --random_seed ${seed} --root_read_path ./${dataset} --root_write_path ../${dataset} --test_epoch 2000 --epoch_num 1000 --dataset_type ${dataset} --train_ratio ${train_ratio} --target_size ${target_size} --area_path ${area_path} --mode test > train_log/inference_${train_flag}.log &
        PID=$!
        wait $PID
    done
done