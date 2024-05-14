import torch
from ptae import PTAE
from config import PTAEConfig, TrainConfig
from data_util import save_latent
import numpy as np
from dataset import PairedNico2BlocksDataset
from torch.utils.data import DataLoader
from data_util import normalise, pad_with_zeros, add_active_feature
from random import choice
from nltk.translate.bleu_score import sentence_bleu
from torch import nn
from proprioception_eval import evaluate

# Find the descriptions via given actions
def main():
    # get the network configuration (parameters such as number of layers and units)
    parameters = PTAEConfig()
    parameters.set_conf("../train/ptae_conf.txt")

    # get the training configuration (batch size, initialisation, number of iterations, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    save_dir = train_conf.save_dir

    # Load the dataset
    training_data = PairedNico2BlocksDataset(train_conf)
    test_data = PairedNico2BlocksDataset(train_conf, True)

    # Get the max and min values for normalisation
    max_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).max()
    min_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).min()
    max_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).max()
    min_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).min()

    # normalise the joint angles, visual features between -1 and 1 and pad the fast actions with zeros
    training_data.B_bw = normalise(training_data.B_bw, max_joint, min_joint) * training_data.B_bin
    training_data.B_fw = normalise(training_data.B_fw, max_joint, min_joint) * training_data.B_bin
    test_data.B_bw = normalise(test_data.B_bw, max_joint, min_joint) * test_data.B_bin
    test_data.B_fw = normalise(test_data.B_fw, max_joint, min_joint) * test_data.B_bin
    training_data.V_bw = normalise(training_data.V_bw, max_vis, min_vis) * training_data.V_bin
    training_data.V_fw = normalise(training_data.V_fw, max_vis, min_vis) * training_data.V_bin
    test_data.V_bw = normalise(test_data.V_bw, max_vis, min_vis) * test_data.V_bin
    test_data.V_fw = normalise(test_data.V_fw, max_vis, min_vis) * test_data.V_bin

    train_dataloader = DataLoader(training_data)
    test_dataloader = DataLoader(test_data)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))
    model = PTAE(parameters, lang_enc_type='None', act_enc_type='None').to(device)

    # Load the trained model
    checkpoint = torch.load(save_dir + '/ptae_regular_unimodal.tar')       # get the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])       # load the model state

    model.eval()

    file = open('../vocabulary.txt', 'r')
    vocab = file.read().splitlines()
    file2 = open('../actions.txt', 'r')
    acts = file2.read().splitlines()
    file1 = open('../arrangements.txt', 'r')
    arrangements = file1.read().splitlines()
    signal = 'describe'
    conflict_type = ['action','colour','speed']
    conflict_type_act = ['action','position','speed']
    actions = ['push', 'pull', 'slide']
    actions_alt = ['move-up', 'move-down', 'move-sideways']
    colours = ['blue','cyan', 'green',  'red', 'violet','yellow']
    colours_alt = ['azure', 'greenish-blue', 'harlequin', 'scarlet', 'purple', 'blonde']
    speeds = ['slowly', 'fast']
    speeds_alt = ['unhurriedly', 'quickly']
    positions = ['l', 'r']
    train_true = 0
    train_false = 0
    test_true = 0
    test_false = 0
    # Feed the dataset as input
    if signal == 'describe' or signal == 'repeat language':
        train_bleu_score = 0
        test_bleu_score = 0

    for input in train_dataloader:
        L_fw_before = input["L_fw"].transpose(0, 1)
        input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
        input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
        input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
        input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
        input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
        input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
        sentence_idx = np.random.randint(8)  # Generate random index for description alternatives
        # Choose one of the eight description alternatives according to the generated random index
        L_fw_feed = L_fw_before[5 * sentence_idx:5 + 5 * sentence_idx, :, :]
        input["L_fw"] = L_fw_feed.to(device)
        L_fw_before = L_fw_before.numpy()
        # introduce conflict in language input
        if signal == 'describe' or signal == 'repeat action':
            if conflict_type.__contains__('action'):
                original_act = vocab[input["L_fw"][1,:,:].argmax(axis=1)]
                if actions.__contains__(original_act):
                    original_act_list_ind = actions.index(original_act)
                    new_act_list_ind = choice([i for i in range(0,len(actions)) if i not in [original_act_list_ind]])
                    input["L_fw"][1, :, :] = nn.functional.one_hot(torch.tensor(vocab.index(actions[new_act_list_ind]), dtype=torch.int64), input['L_fw'].size()[-1]).unsqueeze(0)
                else:
                    original_act_list_ind = actions_alt.index(original_act)
                    new_act_list_ind = choice([i for i in range(0,len(actions_alt)) if i not in [original_act_list_ind]])
                    input["L_fw"][1, :, :] = nn.functional.one_hot(torch.tensor(vocab.index(actions_alt[new_act_list_ind]), dtype=torch.int64), input['L_fw'].size()[-1]).unsqueeze(0)
            if conflict_type.__contains__('colour'):
                original_col = vocab[input["L_fw"][2,:,:].argmax(axis=1)]
                if colours.__contains__(original_col):
                    original_col_list_ind = colours.index(original_col)
                    new_colours_list_ind = choice([i for i in range(0,len(colours)) if i not in [original_col_list_ind]])
                    input["L_fw"][2, :, :] = nn.functional.one_hot(torch.tensor(vocab.index(colours[new_colours_list_ind]), dtype=torch.int64), input['L_fw'].size()[-1]).unsqueeze(0)
                else:
                    original_col_list_ind = colours_alt.index(original_col)
                    new_colours_list_ind = choice([i for i in range(0,len(actions_alt)) if i not in [original_col_list_ind]])
                    input["L_fw"][2, :, :] = nn.functional.one_hot(torch.tensor(vocab.index(colours_alt[new_colours_list_ind]), dtype=torch.int64), input['L_fw'].size()[-1]).unsqueeze(0)
            if conflict_type.__contains__('speed'):
                original_speed = vocab[input["L_fw"][3,:,:].argmax(axis=1)]
                if speeds.__contains__(original_speed):
                    original_speed_list_ind = speeds.index(original_speed)
                    new_speeds_list_ind = choice([i for i in range(0,len(speeds)) if i not in [original_speed_list_ind]])
                    input["L_fw"][3, :, :] = nn.functional.one_hot(torch.tensor(vocab.index(speeds[new_speeds_list_ind]), dtype=torch.int64), input['L_fw'].size()[-1]).unsqueeze(0)
                else:
                    original_speed_list_ind = speeds_alt.index(original_speed)
                    new_speeds_list_ind = choice([i for i in range(0,len(speeds_alt)) if i not in [original_speed_list_ind]])
                    input["L_fw"][3, :, :] = nn.functional.one_hot(torch.tensor(vocab.index(speeds_alt[new_speeds_list_ind]), dtype=torch.int64), input['L_fw'].size()[-1]).unsqueeze(0)
        elif signal == 'repeat language' or 'execute':
            offset = 0
            file_index = int(input['B_filenames'][0].split('target')[-1].split('.txt')[0])# % len(acts)
            if conflict_type_act.__contains__('action'):
                action_type = acts[file_index].split('-')[0]
                original_act_list_ind = actions.index(action_type.swapcase())
                new_idx = choice([i for i in range(0, len(actions)) if i not in [original_act_list_ind]])
                offset = offset + new_idx - original_act_list_ind
            if conflict_type_act.__contains__('position'):
                pos = acts[file_index].split('-')[1]
                if pos == 'L':
                    offset = offset + len(actions)
                else:
                    offset = offset - len(actions)
            if conflict_type_act.__contains__('speed'):
                speed = acts[file_index].split(' ')[0].split('-')[2]
                if speed == 'SLOW':
                    offset = offset + len(acts)/2
                else:
                    offset = offset - len(acts)/2
            new_file_name = input['B_filenames'][0].split('target0')[0] + 'target' + str(file_index+int(offset)).zfill(6) + '.txt'
            if(train_dataloader.dataset.B_filenames.__contains__(new_file_name)):
                input["B_fw"] = torch.tensor(train_dataloader.dataset.B_fw.transpose(1, 0, 2)[train_dataloader.dataset.B_filenames.index(new_file_name), :, :]).unsqueeze(1).to(device)
                input["V_fw"] = torch.tensor(train_dataloader.dataset.V_fw.transpose(1, 0, 2)[train_dataloader.dataset.B_filenames.index(new_file_name), :, :]).unsqueeze(1).to(device)
            else:
                new_file_name = new_file_name.replace('train', 'test')
                input["B_fw"] = torch.tensor(test_dataloader.dataset.B_fw.transpose(1, 0, 2)[test_dataloader.dataset.B_filenames.index(new_file_name), :, :]).unsqueeze(1).to(device)
                input["V_fw"] = torch.tensor(test_dataloader.dataset.V_fw.transpose(1, 0, 2)[test_dataloader.dataset.B_filenames.index(new_file_name), :, :]).unsqueeze(1).to(device)
        with torch.no_grad():
            lang_result, act_result = model.inference_conflict(input, signal)#model.inference(input, signal)#
        lang_result = lang_result.cpu()
        act_result = act_result.cpu()
        save_latent(lang_result.unsqueeze(0), input["L_filenames"][0], "inference")  # save the predicted descriptions

        act_result = (((act_result+1)/2)*(input["max_joint"]-input["min_joint"]))+input["min_joint"]     # get back raw values

        save_latent(act_result.unsqueeze(0), input["B_filenames"][0], "inference")
        r = lang_result.argmax(axis=1).numpy()
        t = L_fw_before[1:5, 0, :].argmax(axis=1)
        t_second = L_fw_before[6:10, 0, :].argmax(axis=1)
        t_third = L_fw_before[11:15, 0, :].argmax(axis=1)
        t_fourth = L_fw_before[16:20, 0, :].argmax(axis=1)
        t_fifth = L_fw_before[21:25, 0, :].argmax(axis=1)
        t_sixth = L_fw_before[26:30, 0, :].argmax(axis=1)
        t_seventh = L_fw_before[31:35, 0, :].argmax(axis=1)
        t_eighth = L_fw_before[36:40, 0, :].argmax(axis=1)

        # cumulative BLEU scores
        # Check if predicted descriptions match the original ones
        if signal == 'describe' or signal == 'repeat language':
            target = [[]]
            prediction = []
            if (r == t).all() or (r == t_second).all() or (r == t_third).all() or (r == t_fourth).all() \
                    or (r == t_fifth).all() or (r == t_sixth).all() or (r == t_seventh).all() or (r == t_eighth).all():
                print(True)
                train_true = train_true + 1
                for k in range(r.size):
                    target[0].append(vocab[t[k]])
                    prediction.append(vocab[r[k]])
                    print(vocab[r[k]], end=" ")
            else:
                print(False)
                print("Expected:", end=" ")
                for k in range(r.size):
                    print(vocab[t[k]], end=" ")
                    target[0].append(vocab[t[k]])
                    prediction.append(vocab[r[k]])
                print()
                print("Produced:", end=" ")
                for k in range(r.size):
                    print(vocab[r[k]], end=" ")
                train_false = train_false + 1
            train_bleu_score = train_bleu_score + sentence_bleu(target, prediction, weights=(1/2, 1/2))
        elif signal == 'execute' or signal == 'repeat action':
            if r[0] == vocab.index('<BOS/EOS>'):
                print(True)
                train_true = train_true + 1
                for k in range(r.size):
                    print(vocab[r[k]], end=" ")
            else:
                print(False)
                print("Expected:", end=" ")
                print(vocab[1], end=" ")
                print()
                train_false = train_false + 1
                print("Produced:", end=" ")
                for k in range(r.size):
                    print(vocab[r[k]], end=" ")
        print()
    print('Training sentence accuracy:', "{0:.2%}".format(train_true / (train_true + train_false)))
    if signal == 'describe' or signal == 'repeat language':
        print('Training BLUE-2 Score:', train_bleu_score/(train_true + train_false))
    # Do the same for the test set
    if train_conf.test:
        print("test!")
        for input in test_dataloader:
            L_fw_before = input["L_fw"].transpose(0, 1)

            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
            input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]

            sentence_idx = np.random.randint(8)  # Generate random index for description alternatives
            # Choose one of the eight description alternatives according to the generated random index
            L_fw_feed = L_fw_before[5 * sentence_idx:5 + 5 * sentence_idx, :, :]
            input["L_fw"] = L_fw_feed.to(device)
            L_fw_before = L_fw_before.numpy()

            if signal == 'describe' or signal == 'repeat action':
                if conflict_type.__contains__('action'):
                    original_act = vocab[input["L_fw"][1, :, :].argmax(axis=1)]
                    if actions.__contains__(original_act):
                        original_act_list_ind = actions.index(original_act)
                        new_act_list_ind = choice(
                            [i for i in range(0, len(actions)) if i not in [original_act_list_ind]])
                        input["L_fw"][1, :, :] = nn.functional.one_hot(
                            torch.tensor(vocab.index(actions[new_act_list_ind]), dtype=torch.int64),
                            input['L_fw'].size()[-1]).unsqueeze(0)
                    else:
                        original_act_list_ind = actions_alt.index(original_act)
                        new_act_list_ind = choice(
                            [i for i in range(0, len(actions_alt)) if i not in [original_act_list_ind]])
                        input["L_fw"][1, :, :] = nn.functional.one_hot(
                            torch.tensor(vocab.index(actions_alt[new_act_list_ind]), dtype=torch.int64),
                            input['L_fw'].size()[-1]).unsqueeze(0)
                if conflict_type.__contains__('colour'):
                    original_col = vocab[input["L_fw"][2, :, :].argmax(axis=1)]
                    if colours.__contains__(original_col):
                        original_col_list_ind = colours.index(original_col)
                        new_colours_list_ind = choice(
                            [i for i in range(0, len(colours)) if i not in [original_col_list_ind]])
                        input["L_fw"][2, :, :] = nn.functional.one_hot(
                            torch.tensor(vocab.index(colours[new_colours_list_ind]), dtype=torch.int64),
                            input['L_fw'].size()[-1]).unsqueeze(0)
                    else:
                        original_col_list_ind = colours_alt.index(original_col)
                        new_colours_list_ind = choice(
                            [i for i in range(0, len(actions_alt)) if i not in [original_col_list_ind]])
                        input["L_fw"][2, :, :] = nn.functional.one_hot(
                            torch.tensor(vocab.index(colours_alt[new_colours_list_ind]), dtype=torch.int64),
                            input['L_fw'].size()[-1]).unsqueeze(0)
                if conflict_type.__contains__('speed'):
                    original_speed = vocab[input["L_fw"][3, :, :].argmax(axis=1)]
                    if speeds.__contains__(original_speed):
                        original_speed_list_ind = speeds.index(original_speed)
                        new_speeds_list_ind = choice(
                            [i for i in range(0, len(speeds)) if i not in [original_speed_list_ind]])
                        input["L_fw"][3, :, :] = nn.functional.one_hot(
                            torch.tensor(vocab.index(speeds[new_speeds_list_ind]), dtype=torch.int64),
                            input['L_fw'].size()[-1]).unsqueeze(0)
                    else:
                        original_speed_list_ind = speeds_alt.index(original_speed)
                        new_speeds_list_ind = choice(
                            [i for i in range(0, len(speeds_alt)) if i not in [original_speed_list_ind]])
                        input["L_fw"][3, :, :] = nn.functional.one_hot(
                            torch.tensor(vocab.index(speeds_alt[new_speeds_list_ind]), dtype=torch.int64),
                            input['L_fw'].size()[-1]).unsqueeze(0)
            elif signal == 'repeat language' or 'execute':
                offset = 0
                file_index = int(input['B_filenames'][0].split('target')[-1].split('.txt')[0]) % len(acts)
                if conflict_type_act.__contains__('action'):
                    action_type = acts[file_index].split('-')[0]
                    original_act_list_ind = actions.index(action_type.swapcase())
                    new_idx = choice([i for i in range(0, len(actions)) if i not in [original_act_list_ind]])
                    offset = offset + new_idx - original_act_list_ind

                if conflict_type_act.__contains__('position'):
                    pos = acts[file_index].split('-')[1]
                    if pos == 'L':
                        offset = offset + len(actions)
                    else:
                        offset = offset - len(actions)
                if conflict_type_act.__contains__('speed'):
                    speed = acts[file_index].split(' ')[0].split('-')[2]
                    if speed == 'SLOW':
                        offset = offset + len(acts) / 2
                    else:
                        offset = offset - len(acts) / 2
                new_file_name = input['B_filenames'][0].split('target0')[0] + 'target' + str(
                    file_index + int(offset)).zfill(6) + '.txt'
                if (test_dataloader.dataset.B_filenames.__contains__(new_file_name)):

                    input["B_fw"] = torch.tensor(test_dataloader.dataset.B_fw.transpose(1, 0, 2)[
                                                 test_dataloader.dataset.B_filenames.index(new_file_name), :, :]).unsqueeze(1).to(device)
                    input["V_fw"] = torch.tensor(test_dataloader.dataset.V_fw.transpose(1, 0, 2)[
                                                 test_dataloader.dataset.B_filenames.index(new_file_name), :, :]).unsqueeze(1).to(device)
                else:
                    new_file_name = new_file_name.replace('test', 'train')
                    input["B_fw"] = torch.tensor(train_dataloader.dataset.B_fw.transpose(1, 0, 2)[
                                                 train_dataloader.dataset.B_filenames.index(new_file_name), :, :]).unsqueeze(1).to(device)
                    input["V_fw"] = torch.tensor(train_dataloader.dataset.V_fw.transpose(1, 0, 2)[
                                                 train_dataloader.dataset.B_filenames.index(new_file_name), :, ]).unsqueeze(1).to(device)
            with torch.no_grad():
                lang_result, act_result = model.inference_conflict(input, signal)
            lang_result = lang_result.cpu()
            act_result = act_result.cpu()
            save_latent(lang_result.unsqueeze(0), input["L_filenames"][0],
                        "inference")  # save the predicted descriptions

            act_result = (((act_result + 1) / 2) * (input["max_joint"] - input["min_joint"])) + input[
                "min_joint"]  # get back raw values

            save_latent(act_result.unsqueeze(0), input["B_filenames"][0], "inference")
            r = lang_result.argmax(axis=1).numpy()
            t = L_fw_before[1:5, 0, :].argmax(axis=1)
            t_second = L_fw_before[6:10, 0, :].argmax(axis=1)
            t_third = L_fw_before[11:15, 0, :].argmax(axis=1)
            t_fourth = L_fw_before[16:20, 0, :].argmax(axis=1)
            t_fifth = L_fw_before[21:25, 0, :].argmax(axis=1)
            t_sixth = L_fw_before[26:30, 0, :].argmax(axis=1)
            t_seventh = L_fw_before[31:35, 0, :].argmax(axis=1)
            t_eighth = L_fw_before[36:40, 0, :].argmax(axis=1)

            # Check if predicted descriptions match the original ones
            if signal == 'describe' or signal == 'repeat language':
                target = [[]]
                prediction = []
                if (r == t).all() or (r == t_second).all() or (r == t_third).all() or (r == t_fourth).all()\
                        or (r == t_fifth).all() or (r == t_sixth).all() or (r == t_seventh).all() or (r == t_eighth).all():
                    print(True)
                    test_true = test_true + 1
                    for k in range(r.size):
                         target[0].append(vocab[t[k]])
                         prediction.append(vocab[r[k]])
                         print(vocab[r[k]], end=" ")
                else:
                    print(False)
                    print("Expected:", end=" ")
                    for k in range(r.size):
                        print(vocab[t[k]], end=" ")
                        target[0].append(vocab[t[k]])
                        prediction.append(vocab[r[k]])
                    print()
                    print("Produced:", end=" ")
                    for k in range(r.size):
                        print(vocab[r[k]], end=" ")
                    test_false = test_false + 1
                test_bleu_score = test_bleu_score + sentence_bleu(target, prediction, weights=(1/2, 1/2))
            elif signal == 'execute' or signal == 'repeat action':
                if r[0] == vocab.index('<BOS/EOS>'):
                    print(True)
                    test_true = test_true + 1
                    for k in range(r.size):
                        print(vocab[r[k]], end=" ")
                else:
                    print(False)
                    print("Expected:", end=" ")
                    print(vocab[1], end=" ")
                    print()
                    print("Produced:", end=" ")
                    for k in range(r.size):
                        print(vocab[r[k]], end=" ")
                    test_false = test_false + 1
            print()
        print('Test sentence accuracy:', "{0:.2%}".format(test_true / (test_true + test_false)))
        if signal == 'describe' or signal == 'repeat language':
            print('Test BLUE-2 Score:', test_bleu_score / (test_true + test_false))
        evaluate(signal)
if __name__ == "__main__":
    main()
