import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from data_util import read_sequential_target
from config import TrainConfig, TrainMTLConfig


def evaluate_ambgs(signal='describe', pre_length=True):
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    B_fw, B_bw, B_bin, B_len, filenames = read_sequential_target(train_conf.B_dir, True)
    B_fw_u, B_bw_u, B_bin_u, B_len_u, filenames_u = read_sequential_target(train_conf.B_dir_test, True)
    max_joint = np.concatenate((B_fw, B_fw_u), 1).max()
    min_joint = np.concatenate((B_fw, B_fw_u), 1).min()

    B_fw = B_fw.transpose((1,0,2))
    B_fw_u = B_fw_u.transpose((1,0,2))
    B_bw = B_bw.transpose((1,0,2))
    B_bw_u = B_bw_u.transpose((1,0,2))
    B_bin = B_bin.transpose((1,0,2))
    B_bin_u = B_bin_u.transpose((1,0,2))
    predict_train, _, predtrain_bin, predtrain_len, _ = read_sequential_target('../train/inference/prediction/behavior_train/', True)
    predict_test, _, predtest_bin, predtest_len, _ = read_sequential_target('../train/inference/prediction/behavior_test/', True)
    predict_train = predict_train.transpose((1,0,2))
    predict_test = predict_test.transpose((1,0,2))
    predtrain_bin = predtrain_bin.transpose((1,0,2))
    predtest_bin = predtest_bin.transpose((1,0,2))
    predict_test = predict_test * predtest_bin[:, :, :]
    if predict_test.shape[1]<(B_fw.shape[1]-1)*2:
        predict_test = np.concatenate((predict_test, np.zeros((predict_test.shape[0],
                                                              (B_fw_u.shape[1]-1)*2-predict_test.shape[1],
                                                              predict_test.shape[2]))), axis=1)
    if signal=='describe':
        if B_bin_u[:,1:,:].shape[1] < predtest_bin.shape[1]:
            gt = np.repeat(np.expand_dims(B_bw_u[:, 0, :], 1), predtest_bin.shape[1], axis=1) * predtest_bin
        else:
            gt = np.repeat(np.expand_dims(B_bw_u[:, 0, :], 1), B_bw_u.shape[1] - 1, axis=1) * B_bin_u[:, 1:, :]
    elif signal == 'execute':
        if B_bin_u[:,1:,:].shape[1] < predtest_bin.shape[1]:
            gt = []
            for i in range(B_fw_u.shape[0]):
                if B_len_u[i] - 1 - predtest_len[i] == 0:
                    gt.append(np.concatenate((B_fw_u[i, 1:, :], np.zeros((B_fw_u.shape[1]-1, B_fw_u.shape[2])))))
                else:
                    static_joints = np.repeat(np.expand_dims(B_fw_u[i, 0, :], 0), B_fw_u.shape[1] - 1, axis=0)
                    gt.append(np.concatenate((static_joints, B_fw_u[i, 1:, :]), axis=0))
            #gt = np.repeat(np.expand_dims(B_fw_u[:, 0, :], 1),  B_fw_u.shape[1] - 1, axis=1) #* predtest_bin
            #gt = np.concatenate((gt, B_fw_u[:,1:,:]), axis=1) * predtest_bin
            gt = np.asarray(gt)
        else:
            gt = B_fw_u[:,1:,:]
    elif signal =='repeat action':
        gt = B_fw_u[:,1:,:]
    else:
        gt = np.repeat(np.expand_dims(B_fw_u[:, 0, :], 1), B_fw_u.shape[1] - 1, axis=1) * B_bin_u[:, 1:, :]
    if pre_length == False:
        #if gt.size()[1] <= predict_test.size()[1]:
        #    predict_test = predict_test[:, :gt.shape[1], :-1] * B_bin_u[:,1:,:]
        if gt.shape[1] > predict_test.shape[1]:
            #predict_test = predict_test[:, :, :-1] * B_bin_u[:,1:predict_test.shape[1] + 1,:]
            predict_test = np.concatenate((predict_test, np.zeros((predict_test.shape[0], gt.shape[1] - predict_test.shape[1],
                                                        predict_test.shape[2]))), 1)
    mse = np.mean(np.square(predict_test - gt))  # action loss (MSE)
    rmse = np.sqrt(mse)
    nrmse = rmse / (max_joint-min_joint)
    #mse = np.mean(np.square(predict_test - B_fw_u[:,1:,:]))
    print('Normalised Root-Mean squared error (NRMSE) for predicted joint values: ',nrmse*100)


def evaluate(signal='describe', pre_length=True):
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    B_fw, B_bw, B_bin, B_len, filenames = read_sequential_target(train_conf.B_dir_val, True)
    B_fw_u, B_bw_u, B_bin_u, B_len_u, filenames_u = read_sequential_target(train_conf.B_dir_test, True)
    max_joint = np.concatenate((B_fw, B_fw_u), 1).max()
    min_joint = np.concatenate((B_fw, B_fw_u), 1).min()
    #min = B_fw.min()
    #max = B_fw.max()
    #min_u = B_fw_u.min()
    #max_u = B_fw_u.max()
    #mean_u = B_fw_u.mean()
    #B_fw = 2 * ((B_fw - B_fw.min())/(B_fw.max()-B_fw.min())) - 1
    #B_fw_u = 2 * ((B_fw_u - B_fw_u.min()) / (B_fw_u.max() - B_fw_u.min())) - 1
    B_fw = B_fw.transpose((1,0,2))
    B_fw_u = B_fw_u.transpose((1,0,2))
    B_bw = B_bw.transpose((1,0,2))
    B_bw_u = B_bw_u.transpose((1,0,2))
    B_bin = B_bin.transpose((1,0,2))
    B_bin_u = B_bin_u.transpose((1,0,2))
    predict_train, _, predtrain_bin, predtrain_len, _ = read_sequential_target('../train/inference/prediction/behavior_train/', True)
    predict_test, _, predtest_bin, predtest_len, _ = read_sequential_target('../train/inference/prediction/behavior_test/', True)
    predict_train = predict_train.transpose((1,0,2))
    predict_test = predict_test.transpose((1,0,2))
    predtrain_bin = predtrain_bin.transpose((1,0,2))
    predtest_bin = predtest_bin.transpose((1,0,2))
    predict_test = predict_test * B_bin_u[:, 1:, :]
    if signal=='describe':
        gt = np.repeat(np.expand_dims(B_bw_u[:,0,:], 1), B_bw_u.shape[1] - 1, axis=1)*B_bin_u[:,1:,:]
    elif signal == 'execute' or signal =='repeat action':
        gt = B_fw_u[:,1:,:]
    else:
        gt = np.repeat(np.expand_dims(B_fw_u[:, 0, :], 1), B_fw_u.shape[1] - 1, axis=1) * B_bin_u[:, 1:, :]
    if pre_length == False:
        #if gt.size()[1] <= predict_test.size()[1]:
        #    predict_test = predict_test[:, :gt.shape[1], :-1] * B_bin_u[:,1:,:]
        if gt.shape[1] > predict_test.shape[1]:
            #predict_test = predict_test[:, :, :-1] * B_bin_u[:,1:predict_test.shape[1] + 1,:]
            predict_test = np.concatenate((predict_test, np.zeros((predict_test.shape[0], gt.shape[1] - predict_test.shape[1],
                                                        predict_test.shape[2]))), 1)
    mse = np.mean(np.square(predict_test - gt))  # action loss (MSE)
    rmse = np.sqrt(mse)
    nrmse = rmse / (max_joint-min_joint)
    #mse = np.mean(np.square(predict_test - B_fw_u[:,1:,:]))
    print('Normalised Root-Mean squared error (NRMSE) for predicted joint values: ',nrmse*100)

def evaluate_mtl(signal='describe', pre_length=True, train_conf=TrainConfig):
    #train_conf = TrainConfig()
    #train_conf.set_conf("../train/train_conf.txt")
    B_fw, B_bw, B_bin, B_len, filenames = read_sequential_target(train_conf.B_dir_val, True, max_len=102)
    B_fw_u, B_bw_u, B_bin_u, B_len_u, filenames_u = read_sequential_target(train_conf.B_dir_test, True, max_len=102)
    max_joint = np.concatenate((B_fw, B_fw_u), 1).max()
    min_joint = np.concatenate((B_fw, B_fw_u), 1).min()
    #min = B_fw.min()
    #max = B_fw.max()
    #min_u = B_fw_u.min()
    #max_u = B_fw_u.max()
    #mean_u = B_fw_u.mean()
    #B_fw = 2 * ((B_fw - B_fw.min())/(B_fw.max()-B_fw.min())) - 1
    #B_fw_u = 2 * ((B_fw_u - B_fw_u.min()) / (B_fw_u.max() - B_fw_u.min())) - 1
    B_fw = B_fw.transpose((1,0,2))
    B_fw_u = B_fw_u.transpose((1,0,2))
    B_bw = B_bw.transpose((1,0,2))
    B_bw_u = B_bw_u.transpose((1,0,2))
    B_bin = B_bin.transpose((1,0,2))
    B_bin_u = B_bin_u.transpose((1,0,2))
    if signal=='describe':
        max_len =None
    else:
        max_len = 102-1
    predict_train, _, predtrain_bin, predtrain_len, _ = read_sequential_target('../train/inference/prediction/behavior_val/', True, max_len=max_len)
    predict_test, _, predtest_bin, predtest_len, _ = read_sequential_target('../train/inference/prediction/behavior_test/', True, max_len=max_len)
    predict_train = predict_train.transpose((1,0,2))
    predict_test = predict_test.transpose((1,0,2))
    predtrain_bin = predtrain_bin.transpose((1,0,2))
    predtest_bin = predtest_bin.transpose((1,0,2))
    if signal=='describe':
        gt = np.expand_dims(B_bw_u[:,0,:], 1)
    elif signal == 'execute' or signal =='repeat action':
        predict_test = predict_test * B_bin_u[:, 1:, :]
        gt = B_fw_u[:,1:,:]
    else:
        gt = np.repeat(np.expand_dims(B_fw_u[:, 0, :], 1), B_fw_u.shape[1] - 1, axis=1) * B_bin_u[:, 1:, :]
    if pre_length == False:
        #if gt.size()[1] <= predict_test.size()[1]:
        #    predict_test = predict_test[:, :gt.shape[1], :-1] * B_bin_u[:,1:,:]
        if gt.shape[1] > predict_test.shape[1]:
            #predict_test = predict_test[:, :, :-1] * B_bin_u[:,1:predict_test.shape[1] + 1,:]
            predict_test = np.concatenate((predict_test, np.zeros((predict_test.shape[0], gt.shape[1] - predict_test.shape[1],
                                                        predict_test.shape[2]))), 1)
    mse = np.mean(np.square(predict_test - gt))  # action loss (MSE)
    rmse = np.sqrt(mse)
    nrmse = rmse / (max_joint-min_joint)
    #mse = np.mean(np.square(predict_test - B_fw_u[:,1:,:]))
    print('Normalised Root-Mean squared error (NRMSE) for predicted joint values: ',nrmse*100)
if __name__ == '__main__':
    train_conf = TrainMTLConfig()
    train_conf.set_conf("../train/train_mtl_conf.txt")
    evaluate_mtl('describe', True, train_conf)
    #evaluate_ambgs('execute', True, train_conf)

