import sys
import torch
import argparse

from tools import function
from config.new_config import ConfigClass
from mynn.neu_protein_trainer import NeuProteinTrainer

def get_args():
    parser = argparse.ArgumentParser(description="HyperParameters for Sequence Embedding")

    # loss and model
    parser.add_argument("--network_type",                   type=str,   default="MYCNN", help="network type")
    parser.add_argument("--loss_type",                      type=str,   default="triplet", help="loss type")
    parser.add_argument("--cnn_feature_distance_type",      type=str,   default="euclidean_sep", help="cnn feature distance type")
    parser.add_argument("--cnntotal_feature_distance_type", type=str,   default="euclidean", help="@Deprecated")
    parser.add_argument("--all_feature_distance_type",      type=str,   default="euclidean", help="all feature distance type")
    parser.add_argument("--sampling_type",                  type=str,   default="distance_sampling3", help="sampling type")
    parser.add_argument("--train_flag",                     type=str,   default="test", help="train flag")
    parser.add_argument("--head_num",                       type=str,   default=1, help="mlp head num")
    parser.add_argument("--area_path",                      type=str,   default="", help="selection area path")
    
    
    # hyperpara
    parser.add_argument("--target_size",                    type=int,   default=128, help="mlp target size")
    parser.add_argument("--channel",                        type=int,   default=8, help="channel num")
    parser.add_argument("--sampling_num",                   type=int,   default=1, help="sampling num for each sampling type")
    parser.add_argument("--epoch_num",                      type=int,   default=1000, help="epoch num")
    parser.add_argument("--device",                         type=str,   default="cuda:0", help="device")
    
    parser.add_argument("--learning_rate",                  type=float, default=0.001, help="learning rate")
    parser.add_argument("--train_ratio",                    type=float, default=1, help="train ratio")
    parser.add_argument("--batch_size",                     type=int,   default=128, help="batch size")
    parser.add_argument("--random_seed",                    type=int,   default=666, help="random seed")
    parser.add_argument("--mode",                           type=str,   default="train-directly", help="mode")
    parser.add_argument("--test_epoch",                     type=int,   default=5, help="test epoch")
    parser.add_argument("--print_epoch",                    type=int,   default=1, help="print epoch")
    parser.add_argument("--save_model",                     type=bool,  default=False, help="save model")
    parser.add_argument("--save_model_epoch",               type=int,   default=5, help="save model epoch")

    parser.add_argument("--root_write_path",                type=str,   default="/mnt/sda/czh/Neuprotein/5000_needle_512", help="root write path")
    parser.add_argument("--root_read_path",                type=str,   default="/mnt/sda/czh/Neuprotein/5000_needle_512", help="root read path")
    parser.add_argument("--similarity_type",                type=str,   default="needle", help="similarity type")
    parser.add_argument("--dataset_type",                   type=str,   default="protein", help="dataset type")

    args = parser.parse_args()

    return args
    

if __name__ == "__main__":
    
    if not torch.cuda.is_available():
        print("GPU is not available.")
        exit()

    args = get_args()
    device = torch.device(args.device)
    print("Device is:", device)

    # set random seed
    function.setup_seed(args.random_seed)

    # split dataset
    train_set, query_set, base_set = function.set_dataset(args.root_read_path)

    # set different input dim of dataset
    if args.dataset_type == "big_uniref":
        input_size = 24
    elif args.dataset_type == "big_uniprot":
        input_size = 25
    elif args.dataset_type == "protein":
        input_size = 20
    elif args.dataset_type == "dna":
        input_size = 4
    elif args.dataset_type == "big_qiita":
        input_size = 4
    elif args.dataset_type == "big_rt988":
        input_size = 4
    elif args.dataset_type == "big_gen50ks":
        input_size = 4
    else:
        raise ValueError("dataset_type error")

    save_model_path = args.root_write_path + "/model"

    my_config = ConfigClass(input_size                     = input_size,
                            target_size                    = args.target_size,
                            batch_size                     = args.batch_size,
                            sampling_num                   = args.sampling_num,
                            learning_rate                  = args.learning_rate,
                            epoch_num                      = args.epoch_num,
                            network_type                   = args.network_type,
                            channel                        = args.channel,
                            loss_type                      = args.loss_type,
                            cnn_feature_distance_type      = args.cnn_feature_distance_type,
                            cnntotal_feature_distance_type = args.cnntotal_feature_distance_type,
                            all_feature_distance_type      = args.all_feature_distance_type,
                            sampling_type                  = args.sampling_type,
                            root_write_path                = args.root_write_path,
                            root_read_path                = args.root_read_path,
                            train_ratio                    = args.train_ratio,
                            mode                           = args.mode,
                            test_epoch                     = args.test_epoch,
                            print_epoch                    = args.print_epoch,
                            save_model                     = args.save_model,
                            save_model_path                = save_model_path,
                            similarity_type                = args.similarity_type,
                            dataset_type                   = args.dataset_type,
                            device                         = device,
                            LDS                            = False,
                            FDS                            = False,
                            train_flag                     = args.train_flag,
                            head_num                       = int(args.head_num),
                            area_path                      = args.area_path,
                            train_set                      = train_set,
                            query_set                      = query_set,
                            base_set                       = base_set)

    protein_network = NeuProteinTrainer(my_config)

    vec_list_path          = my_config.my_dict["root_read_path"] + '/onehot_list'
    num_list_path          = my_config.my_dict["root_read_path"] + '/num_list'
    seq_list_path          = my_config.my_dict["root_read_path"] + '/seq_list'
    onehot_list_path       = my_config.my_dict["root_read_path"] + '/onehot_list'

    # set similarity matrix path
    if args.similarity_type == "needle":
        train_similarity_matrix_path = my_config.my_dict["root_read_path"] + '/train_similarity_matrix_result'
        test_similarity_matrix_path  = my_config.my_dict["root_read_path"] + '/test_similarity_matrix_result'
        train_aln_alp_path           = my_config.my_dict["root_read_path"] + '/train_aln_alp_matrix_result2'
    elif args.similarity_type == "ed":
        train_similarity_matrix_path = my_config.my_dict["root_read_path"] + '/train_ed_distance_matrix_result'
        test_similarity_matrix_path  = my_config.my_dict["root_read_path"] + '/test_ed_distance_matrix_result'
        train_aln_alp_path           = my_config.my_dict["root_read_path"] + '/train_ed_aln_alp_matrix_result2'
    else:
        raise ValueError("similarity_type error")
    
    train_vec_path         = my_config.my_dict["root_read_path"] + '/1train_onehot_list'
    train_num_path         = my_config.my_dict["root_read_path"] + '/1train_num_list'
    train_length_path      = my_config.my_dict["root_read_path"] + '/1train_length_list'
    train_similarity_path  = my_config.my_dict["root_read_path"] + '/1train_similarity_result'


    protein_network.data_prepare(vec_list_path,
                                 num_list_path,
                                 train_similarity_matrix_path,
                                 train_aln_alp_path,
                                 test_similarity_matrix_path)

    mode = my_config.my_dict["mode"]
    if mode == "test":
        # extract feature from model
        protein_network.extract_feature_from_path()
    elif mode == "train-directly":
        # directly train of old method
        protein_network.train()
    elif mode == "multihead":
        # directly train of our multihead
        protein_network.train_multi_head()
    elif mode == "multihead-inference":
        # extract feature from our multihead model
        protein_network.extract_feature_from_path_multi_head()
    else:
        raise ValueError("Train Mode Value Error!")

    