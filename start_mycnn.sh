train_ratio=1
target_size=128
batch_size=128
dataset_list=(big_uniref)
seed_list=(666 555 444)
area_path_list=(agglomerative_1000_100_2_None)
for dataset in ${dataset_list[@]}; do
    for seed in ${seed_list[@]}; do
        for area in ${area_path_list[@]}; do
            area_path=./${dataset}/clustering/${area}
            train_flag=MYCNN_EqualSize_${dataset}_${seed}_${area}_${target_size}

            echo Training_${train_flag}
            nohup python train.py --train_flag ${train_flag} --random_seed ${seed} --root_read_path ./${dataset} --root_write_path ./${dataset} --test_epoch 2000 --dataset_type ${dataset}  --mode multihead --target_size ${target_size} --train_ratio ${train_ratio} --batch_size ${batch_size} --area_path ${area_path} > train_log/training_${train_flag}.log &
            PID0=$!
            wait $PID0
            
            echo Inference_${train_flag}
            nohup python train.py --train_flag ${train_flag} --random_seed ${seed} --root_read_path ./${dataset} --root_write_path ./${dataset} --test_epoch 2000 --dataset_type ${dataset}  --mode multihead-inference --target_size ${target_size} --train_ratio ${train_ratio} --batch_size ${batch_size} --area_path ${area_path} > train_log/inference_${train_flag}.log &
            PID0=$!
            wait $PID0
        done
    done
done
