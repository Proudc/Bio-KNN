import os
import json
import numpy as np

import torch

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, type(bytes)):
            return str(obj, encoding='utf-8')
        elif type(obj) == int:
            return str(obj)
        elif type(obj) == float:
            return str(obj)
        elif type(obj) == bool:
            return str(obj)
        elif type(obj) == torch.device:
            return str(obj)
        else:
            print(type(obj))
            return super(MyEncoder, self).default(obj)

class ConfigClass():
    def __init__(self,
                 input_size                     = None,
                 target_size                    = None,
                 batch_size                     = None,
                 sampling_num                   = None,
                 learning_rate                  = None,
                 epoch_num                      = None,
                 network_type                   = None,
                 channel                        = None,
                 loss_type                      = None,
                 cnn_feature_distance_type      = None,
                 cnntotal_feature_distance_type = None,
                 all_feature_distance_type      = None,
                 sampling_type                  = None,
                 root_write_path                = None,
                 root_read_path                = None,
                 train_ratio                    = None,
                 mode                           = None,
                 test_epoch                     = None,
                 print_epoch                    = None,
                 save_model                     = None,
                 save_model_path                = None,
                 similarity_type                = None,
                 dataset_type                   = None,           
                 device                         = "cpu",
                 LDS                            = False,
                 FDS                            = True,
                 train_flag                     = "train.log",
                 head_num                       = None,
                 area_path                      = None,
                 train_set                      = None,
                 query_set                      = None,
                 base_set                       = None):
        self.my_dict = {}
        self.my_dict["input_size"]                     = input_size
        self.my_dict["target_size"]                    = target_size
        self.my_dict["batch_size"]                     = batch_size
        self.my_dict["sampling_num"]                   = sampling_num
        self.my_dict["learning_rate"]                  = learning_rate
        self.my_dict["epoch_num"]                      = epoch_num
        self.my_dict["network_type"]                   = network_type
        self.my_dict["channel"]                        = channel
        self.my_dict["loss_type"]                      = loss_type
        self.my_dict["cnn_feature_distance_type"]      = cnn_feature_distance_type
        self.my_dict["cnntotal_feature_distance_type"] = cnntotal_feature_distance_type
        self.my_dict["all_feature_distance_type"]      = all_feature_distance_type
        self.my_dict["sampling_type"]                  = sampling_type
        self.my_dict["root_write_path"]                = root_write_path
        self.my_dict["root_read_path"]                = root_read_path
        self.my_dict["train_ratio"]                    = train_ratio
        self.my_dict["mode"]                           = mode
        self.my_dict["test_epoch"]                     = test_epoch
        self.my_dict["print_epoch"]                    = print_epoch
        self.my_dict["save_model"]                     = save_model
        self.my_dict["save_model_path"]                = save_model_path
        self.my_dict["similarity_type"]                = similarity_type
        self.my_dict["dataset_type"]                   = dataset_type
        self.my_dict["device"]                         = device
        self.my_dict["LDS"]                            = LDS
        self.my_dict["FDS"]                            = FDS
        self.my_dict["train_flag"]                     = train_flag
        self.my_dict["head_num"]                       = head_num
        self.my_dict["area_path"]                      = area_path
        self.my_dict["train_set"]                      = train_set
        self.my_dict["query_set"]                      = query_set
        self.my_dict["base_set"]                       = base_set
        self.write_config_to_file()
        
    def write_config_to_file(self):
        if not os.path.exists(self.my_dict["root_write_path"] + "/train_config/"):
            os.mkdir(self.my_dict["root_write_path"] + "/train_config/")
        with open(self.my_dict["root_write_path"] + "/train_config/" + self.my_dict["train_flag"] + ".json", "w", encoding = "utf-8") as f:
            json.dump(self.my_dict, f, ensure_ascii = False, cls = MyEncoder)
            

    def read_config_from_file(self):
        self.my_dict = {}
        with open(self.my_dict["root_write_path"] + "/train_config/" + self.my_dict["train_flag"] + ".json", "r", encoding = "utf-8") as f:
            self.my_dict = json.load(f)
