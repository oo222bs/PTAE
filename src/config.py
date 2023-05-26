#! -*- coding:utf-8 -*-

# Network configuration class
class NetConfig():
    def __init__(self):
        self.L_num_units = 10
        self.L_num_layers = 2
        self.VB_num_units = 10
        self.VB_num_layers = 2
        self.hidden_dim = 10
        self.delta = 1.0
        self.L_weight = 1.0
        self.B_weight = 1.0
        self.S_weight = 1.0
        
        self._int = ["L_num_units", "L_num_layers",
                     "VB_num_units", "VB_num_layers",
                     "hidden_dim"]
        self._float = ["delta",
                       "L_weight",
                       "B_weight",
                       "S_weight"]
        
    def _setattr(self, name, value):
        if name in self._int:
            value = int(value)
            setattr(self, name, value)
        elif name in self._float:
            value = float(value)
            setattr(self, name, value)
        else:
            print("{} can not be changed".format(name))
                
    def _set_param(self, name, value):
        if hasattr(self, name):
            self._setattr(name, value)
        else:
            "{} does not exists!".format(name)
        
    def set_conf(self, conf_file):
        f = open(conf_file, "r")
        line = f.readline()[:-1]
        while line:
            key, value = line.split(": ")
            self._set_param(key, value)
            line = f.readline()[:-1]


# VAE configuration class
class VaeConfig():
    def __init__(self):
        self.L_input_dim = 23
        self.L_num_units = 10
        self.L_num_layers = 2
        self.VB_input_dim = 35
        self.VB_num_units = 10
        self.VB_num_layers = 2
        self.B_input_dim = 5
        self.hidden_dim = 10
        self.delta = 1.0
        self.L_weight = 1.0
        self.B_weight = 1.0
        self.S_weight = 1.0
        self.KL_weight = 1.0

        self._int = ["L_input_dim", "L_num_units", "L_num_layers",
                     "VB_input_dim", "VB_num_units", "VB_num_layers",
                     "B_input_dim", "hidden_dim"]
        self._float = ["delta",
                       "L_weight",
                       "B_weight",
                       "S_weight", "KL_weight"]

    def _setattr(self, name, value):
        if name in self._int:
            value = int(value)
            setattr(self, name, value)
        elif name in self._float:
            value = float(value)
            setattr(self, name, value)
        else:
            print("{} can not be changed".format(name))

    def _set_param(self, name, value):
        if hasattr(self, name):
            self._setattr(name, value)
        else:
            "{} does not exists!".format(name)

    def set_conf(self, conf_file):
        f = open(conf_file, "r")
        line = f.readline()[:-1]
        while line:
            key, value = line.split(": ")
            self._set_param(key, value)
            line = f.readline()[:-1]

# PTAE configuration class
class PtaeConfig():
    def __init__(self):
        self.L_input_dim = 23
        self.L_num_units = 10
        self.L_num_layers = 2
        self.L_max_length = 10
        self.VB_input_dim = 35
        self.VB_num_units = 10
        self.VB_num_layers = 2
        self.B_input_dim = 5
        self.B_max_length = 100
        self.hidden_dim = 10
        self.L_weight = 1.0
        self.B_weight = 1.0
        self.T_num_heads = 8
        self.T_num_layers = 6
        self.T_dropout = 0.1
        self.T_ff_dim = 2048
        self.num_signals = 5

        self._int = ["L_input_dim", "L_num_units", "L_num_layers", "L_max_length",
                     "VB_input_dim", "VB_num_units", "VB_num_layers",
                     "B_input_dim", "B_max_length", "hidden_dim", "T_num_heads", "T_num_layers", "T_ff_dim", "num_signals"]
        self._float = ["L_weight",
                       "B_weight",
                       "T_dropout"]

    def _setattr(self, name, value):
        if name in self._int:
            value = int(value)
            setattr(self, name, value)
        elif name in self._float:
            value = float(value)
            setattr(self, name, value)
        else:
            print("{} can not be changed".format(name))

    def _set_param(self, name, value):
        if hasattr(self, name):
            self._setattr(name, value)
        else:
            "{} does not exists!".format(name)

    def set_conf(self, conf_file):
        f = open(conf_file, "r")
        line = f.readline()[:-1]
        while line:
            key, value = line.split(": ")
            self._set_param(key, value)
            line = f.readline()[:-1]

# XBit configuration class
class XBitConfig():
    def __init__(self):
        self.L_input_dim = 23
        self.L_num_units = 10
        self.L_num_layers = 2
        self.L_max_length = 10
        self.VB_input_dim = 35
        self.VB_num_units = 10
        self.VB_num_layers = 2
        self.B_input_dim = 5
        self.B_max_length = 100
        self.B_binary_dim = 1
        self.hidden_dim = 10
        self.resnet_version = 18
        self.L_weight = 1.0
        self.B_weight = 1.0
        self.T_num_heads = 8
        self.T_num_layers = 6
        self.T_dropout = 0.1
        self.T_ff_dim = 2048
        self.num_signals = 2

        self._int = ["L_input_dim", "L_num_units", "L_num_layers", "L_max_length",
                     "VB_input_dim", "VB_num_units", "VB_num_layers",
                     "B_input_dim", "B_max_length", "B_binary_dim", "hidden_dim", "resnet_version", "T_num_heads", "T_num_layers", "T_ff_dim",  "num_signals"]
        self._float = ["L_weight",
                       "B_weight",
                       "T_dropout"]

    def _setattr(self, name, value):
        if name in self._int:
            value = int(value)
            setattr(self, name, value)
        elif name in self._float:
            value = float(value)
            setattr(self, name, value)
        else:
            print("{} can not be changed".format(name))

    def _set_param(self, name, value):
        if hasattr(self, name):
            self._setattr(name, value)
        else:
            "{} does not exists!".format(name)

    def set_conf(self, conf_file):
        f = open(conf_file, "r")
        line = f.readline()[:-1]
        while line:
            key, value = line.split(": ")
            self._set_param(key, value)
            line = f.readline()[:-1]

# Training configuration class
class TrainConfig():
    def __init__(self):
        self.seed = None
        self.test = 0
        self.num_of_iterations = 100
        self.num_of_epochs = 10
        self.log_interval = 10
        self.test_interval = 10
        self.learning_rate = 0.001
        self.batch_size = 10
        self.noise_std = 0.0
        self.L_dir = "./data/"
        self.B_dir = "./data/"
        self.V_dir = "./data/"
        self.V_opp_dir = "./data/"
        self.IM_dir = "./data/"
        self.L_dir_test = None
        self.B_dir_test = None
        self.V_dir_test = None
        self.V_opp_dir_test =None
        self.IM_dir_test = None
        self.feature_dir = None
        self.save_dir = "./checkpoints/"
        self.cae_save_dir = "./cae_checkpoints/"
        self.cae_save_dir_red = "./cae_red_checkpoints/"
        self.cae_save_dir_green = "./cae_green_checkpoints/"
        self.cae_save_dir_blue = "./cae_blue_checkpoints/"
        
    def _setattr(self, name, value):
        if name in ["seed", "test", "num_of_iterations",
                    "num_of_epochs", "log_interval", "test_interval",
                    "batch_size"]:
            value = int(value)
        if name in ["learning_rate", "noise_std"]:
            value = float(value)
        setattr(self, name, value)
            
    def _set_param(self, name, value):
        if hasattr(self, name):
            self._setattr(name, value)
        else:
            "{} does not exists!".format(name)
        
    def set_conf(self, conf_file):
        f = open(conf_file, "r")
        line = f.readline()[:-1]
        while line:
            key, value = line.split(": ")
            self._set_param(key, value)
            line = f.readline()[:-1]


# Training configuration class
class TrainMTLConfig():
    def __init__(self):
        self.seed = None
        self.test = 0
        self.num_of_iterations = 100
        self.num_of_epochs = 10
        self.log_interval = 10
        self.test_interval = 10
        self.learning_rate = 0.001
        self.batch_size = 10
        self.noise_std = 0.0
        self.L_dir = "./data/"
        self.L_oh_dir = "./data/"
        self.B_dir = "./data/"
        self.P_dir = "./data/"
        self.P_out_dir = "./data/"
        self.IM_dir = "./data/"
        self.IM_left_dir = "./data/"
        self.IM_right_dir = "./data/"
        self.IM_wrist_dir = "./data/"
        self.L_dir_val = None
        self.L_oh_dir_val = None
        self.B_dir_val = None
        self.P_dir_val = None
        self.P_out_dir_val = None
        self.IM_dir_val = None
        self.IM_left_dir_val = None
        self.IM_right_dir_val = None
        self.IM_wrist_dir_val = None
        self.L_dir_test = None
        self.L_oh_dir_test = None
        self.B_dir_test = None
        self.P_dir_test = None
        self.P_out_dir_test = None
        self.IM_dir_test = None
        self.IM_left_dir_test = None
        self.IM_right_dir_test = None
        self.IM_wrist_dir_test = None
        self.feature_dir = None
        self.save_dir = "./checkpoints/"
        self.cae_save_dir = "./cae_checkpoints/"
        self.cae_save_dir_red = "./cae_red_checkpoints/"
        self.cae_save_dir_green = "./cae_green_checkpoints/"
        self.cae_save_dir_blue = "./cae_blue_checkpoints/"

    def _setattr(self, name, value):
        if name in ["seed", "test", "num_of_iterations",
                    "num_of_epochs", "log_interval", "test_interval",
                    "batch_size"]:
            value = int(value)
        if name in ["learning_rate", "noise_std"]:
            value = float(value)
        setattr(self, name, value)

    def _set_param(self, name, value):
        if hasattr(self, name):
            self._setattr(name, value)
        else:
            "{} does not exists!".format(name)

    def set_conf(self, conf_file):
        f = open(conf_file, "r")
        line = f.readline()[:-1]
        while line:
            key, value = line.split(": ")
            self._set_param(key, value)
            line = f.readline()[:-1]
