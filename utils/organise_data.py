from config import NetConfig, TrainConfig
from data_util import read_sequential_target
import os
import numpy as np
from PIL import Image
import re
import csv


def organise_joint_angles(logs_path):
    file2 = open('../actionstcnoexc.txt', 'r')
    acts = file2.read().splitlines()
    test = []
    for i, act in enumerate(acts):
        acts[i] = act.split(' ')[0]
        if act.split(' ')[-1] == "test":
            test.append(i)
    all_file_list = []
    dir_list = []
    for (root, dirs, files) in os.walk(logs_path):
        dir_list.append(root)
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        file_list.sort()
        if(len(file_list)>0):
            all_file_list.append(file_list)

    all_file_list.sort()
    max_len = 1
    count = 0
    #all_resized_images = np.zeros((len(all_file_list), max_len, 120, 160, 1))
    #test = [2, 6, 9, 15, 19, 22, 24, 28, 35, 37, 41, 44, 50, 54, 57, 63, 67, 70,
            #72, 76, 83, 85, 89, 92, 98, 102, 105, 111, 115, 118, 120, 124, 131, 133, 137, 140]
    for file_list in all_file_list:
        num_of_images = len(file_list)
        file_list.sort()
        for i, filename in enumerate(file_list):
            date='220420'
            rows = []
            with open(filename, 'r') as file:
                csvreader = csv.reader(file)
                header = next(csvreader)
                for row in csvreader:
                    rows.append(row)

            action = filename.split(os.path.sep)[-1].split('.')[0]
            if action == 'push_left':
                indices = [i for i, x in enumerate(acts) if x == "PUSH-L"]
            elif action == 'pull_left':
                indices = [i for i, x in enumerate(acts) if x == "PULL-L"]
            elif action == 'shift_left_to_left':
                indices = [i for i, x in enumerate(acts) if x == "SLIDE-L-L"]
            elif action == 'shift_left_to_right':
                indices = [i for i, x in enumerate(acts) if x == "SLIDE-L-R"]
            elif action == 'push_mid':
                indices = [i for i, x in enumerate(acts) if x == "PUSH-M"]
            elif action == 'pull_mid':
                indices = [i for i, x in enumerate(acts) if x == "PULL-M"]
            elif action == 'shift_mid_to_left':
                indices = [i for i, x in enumerate(acts) if x == "SLIDE-M-L"]
            elif action == 'shift_mid_to_right':
                indices = [i for i, x in enumerate(acts) if x == "SLIDE-M-R"]
            elif action == 'push_right':
                indices = [i for i, x in enumerate(acts) if x == "PUSH-R"]
            elif action == 'pull_right':
                indices = [i for i, x in enumerate(acts) if x == "PULL-R"]
            elif action == 'shift_right_to_left':
                indices = [i for i, x in enumerate(acts) if x == "SLIDE-R-L"]
            elif action == 'shift_right_to_right':
                indices = [i for i, x in enumerate(acts) if x == "SLIDE-R-R"]

            print('test')
            for index in indices:
                save_name_v = "target" + str(index).zfill(4) + '.txt'
                joints = np.array(rows)
                if index in test:
                    train_test = "behavior_test/"
                else:
                    train_test = "behavior_train/"
                dirname_v = '../target_three_cubes_no_exc/' + train_test + date + "/"
                # save_name = os.path.join(dirname, save_name)
                save_name_t = os.path.join(dirname_v, save_name_v)
                if not os.path.exists(os.path.dirname(save_name_t)):
                    os.makedirs(os.path.dirname(save_name_t))
                np.savetxt(save_name_t, joints, '%s')

def organise_images_three_cubes(images_path):
    file1 = open('../arrangementstcnoexc.txt', 'r')
    arrangements = file1.read().splitlines()
    file2 = open('../actionstcnoexc.txt', 'r')
    acts = file2.read().splitlines()
    all_file_list = []
    dir_list = []
    for (root, dirs, files) in os.walk(images_path):
        dir_list.append(root)
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        file_list.sort()
        if(len(file_list)>0):
            all_file_list.append(file_list)

    all_file_list.sort()
    max_len = 1
    count = 0
    #all_resized_images = np.zeros((len(all_file_list), max_len, 120, 160, 1))
    for file_list in all_file_list:
        num_of_images = len(file_list)
        file_list.sort()
        resized_images = []
        for i, image in enumerate(file_list):
            date='220420'
            arrange = re.sub(r"([A-Z])", r" \1", image.split(os.path.sep)[-2].split('_')[0]).split()
            arrange = arrange[0].lower() + '+' + arrange[1].lower() + '+' +arrange[2].lower()
            action = image.split(os.path.sep)[-2].split('_', 1)[-1]
            if action == 'push_left':
                act_offset = 0
            elif action == 'pull_left':
                act_offset = 1
            elif action == 'shift_left_to_left':
                act_offset = 2
            elif action == 'shift_left_to_right':
                act_offset = 3
            elif action == 'push_mid':
                act_offset = 4
            elif action == 'pull_mid':
                act_offset = 5
            elif action == 'shift_mid_to_left':
                act_offset = 6
            elif action == 'shift_mid_to_right':
                act_offset = 7
            elif action == 'push_right':
                act_offset = 8
            elif action == 'pull_right':
                act_offset = 9
            elif action == 'shift_right_to_left':
                act_offset = 10
            elif action == 'shift_right_to_right':
                act_offset = 11
            target_no = arrangements.index(arrange) + act_offset
            save_name_v = "target" + str(target_no).zfill(4)
            #left_right = image.split(os.path.sep)[-2]
            original_image = Image.open(image)
            dirname_v = '../target_three_cubes_no_exc/' + "image_train/" + date + "/"
            image_name = str(image.split(os.path.sep)[-1].split('.')[0]).zfill(3) + '.png'
            # save_name = os.path.join(dirname, save_name)
            save_name_t = os.path.join(dirname_v, save_name_v + '/' + image_name)#save_name_t = os.path.join(dirname_v, save_name_v + '/' + left_right+'/'+ image_name)
            if not os.path.exists(os.path.dirname(save_name_t)):
                os.makedirs(os.path.dirname(save_name_t))
            # np.savetxt(save_name, behavior[i], fmt="%.6f")
            original_image.save(save_name_t)
        count = count+1
    return all_file_list


def organise_images_train_test(images_path, offset):
    all_file_list = []
    dir_list = []
    for (root, dirs, files) in os.walk(images_path):
        dir_list.append(root)
        file_list = []
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_list.append(os.path.join(root, file))
        file_list.sort()
        if(len(file_list)>0):
            all_file_list.append(file_list)

    all_file_list.sort()
    max_len = 1
    count = 0
    date = ['201207', '201223', '210107', '210115', '210129', '210203']
    #all_resized_images = np.zeros((len(all_file_list), max_len, 120, 160, 1))
    for file_list in all_file_list:
        num_of_images = len(file_list)
        file_list.sort()
        resized_images = []
        for i, image in enumerate(file_list):
            save_name_v = "target" + str(int(image.split('/')[-2])+offset).zfill(6)
            original_image = Image.open(image)
            #dirname_v = '../target_RL_updated/' + "image_train/" + date + "/"
            dirname_v = '../target_RL_updated/' + "image_train_unlbld/" + date + "/"
            # save_name = os.path.join(dirname, save_name)
            save_name_t = os.path.join(dirname_v, save_name_v + '/' + image.split(os.path.sep)[-1])
            if not os.path.exists(os.path.dirname(save_name_t)):
                os.makedirs(os.path.dirname(save_name_t))
            # np.savetxt(save_name, behavior[i], fmt="%.6f")
            original_image.save(save_name_t)
        count = count+1
    return all_file_list

def sort_by_int(e):
    return int(e.split('.')[0])

def organise_images(images_path):
    all_file_list = []
    dir_list = []
    for (root, dirs, files) in os.walk(images_path):
        dir_list.append(root)
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        file_list.sort()
        if(len(file_list)>0):
            all_file_list.append(file_list)

    all_file_list.sort()
    max_len = 1
    count = 0
    #all_resized_images = np.zeros((len(all_file_list), max_len, 120, 160, 1))
    for file_list in all_file_list:
        num_of_images = len(file_list)
        file_list.sort()
        resized_images = []
        for i, image in enumerate(file_list):
            if i<600:
                date = '201207'
            elif i<1200:
                date = '201223'
            elif i<1800:
                date = '210107'
            elif i<2400:
                date = '210115'
            elif i<3000:
                date ='210129'
            else:
                date ='210203'

            if i<100 or 600<=i<700 or 1200<=i<1300 or 1800<=i<1900 or 2400<=i<2500 or 3000<=i<3100:
                save_name_v = "target000066"
            elif 100<=i<200 or 700 <= i < 800 or 1300 <= i < 1400 or 1900 <= i < 2000 or 2500 <= i < 2600 or 3100 <= i < 3200:
                save_name_v = "target000067"
            elif 200<=i<300 or 800 <= i < 900 or 1400 <= i < 1500 or 2000 <= i < 2100 or 2600 <= i < 2700 or 3200 <= i < 3300:
                save_name_v = "target000068"
            elif 300<=i<400 or 900 <= i < 1000 or 1500 <= i < 1600 or 2100 <= i < 2200 or 2700 <= i < 2800 or 3300 <= i < 3400:
                save_name_v = "target000069"
            elif 400<=i<500 or 1000 <= i < 1100 or 1600 <= i < 1700 or 2200 <= i < 2300 or 2800 <= i < 2900 or 3400 <= i < 3500:
                save_name_v = "target000070"
            else:
                save_name_v = "target000071"
            original_image = Image.open(image)
            dirname_v = '../target/' + "image_train/" + date + "/"
            # save_name = os.path.join(dirname, save_name)
            save_name_t = os.path.join(dirname_v, save_name_v + '/' + image.split(os.path.sep)[-1])
            if not os.path.exists(os.path.dirname(save_name_t)):
                os.makedirs(os.path.dirname(save_name_t))
            # np.savetxt(save_name, behavior[i], fmt="%.6f")
            original_image.save(save_name_t)
            if i % 2 == 0:
                t = int(save_name_v.split('target')[-1])
                save_name_v = "target" + str(t + 72).zfill(6) + '/' + image.split(os.path.sep)[-1]
                dirname_v = '../target/' + "image_train/" + date + "/"
                #save_name = os.path.join(dirname, save_name)
                save_name_t = os.path.join(dirname_v, save_name_v)
                if not os.path.exists(os.path.dirname(save_name_t)):
                    os.makedirs(os.path.dirname(save_name_t))
                #np.savetxt(save_name, behavior[i], fmt="%.6f")
                original_image.save(save_name_t)
        count = count+1
    return all_file_list

def organise_images_opp_agent(images_path):
    all_file_list = []
    dir_list = []
    for (root, dirs, files) in os.walk(images_path):
        dir_list.append(root)
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        file_list.sort()
        if(len(file_list)>0):
            all_file_list.append(file_list)

    all_file_list.sort()
    max_len = 1
    count = 0
    #all_resized_images = np.zeros((len(all_file_list), max_len, 120, 160, 1))
    for file_list in all_file_list:
        num_of_images = len(file_list)
        file_list.sort()
        resized_images = []
        for i, image in enumerate(file_list):
            if i<600:
                date = '201207'
            elif i<1200:
                date = '201223'
            elif i<1800:
                date = '210107'
            elif i<2400:
                date = '210115'
            elif i<3000:
                date ='210129'
            else:
                date ='210203'

            if i<100 or 600<=i<700 or 1200<=i<1300 or 1800<=i<1900 or 2400<=i<2500 or 3000<=i<3100:
                save_name_v = "target000069"
            elif 100<=i<200 or 700 <= i < 800 or 1300 <= i < 1400 or 1900 <= i < 2000 or 2500 <= i < 2600 or 3100 <= i < 3200:
                save_name_v = "target000070"
            elif 200<=i<300 or 800 <= i < 900 or 1400 <= i < 1500 or 2000 <= i < 2100 or 2600 <= i < 2700 or 3200 <= i < 3300:
                save_name_v = "target000071"
            elif 300<=i<400 or 900 <= i < 1000 or 1500 <= i < 1600 or 2100 <= i < 2200 or 2700 <= i < 2800 or 3300 <= i < 3400:
                save_name_v = "target000066"
            elif 400<=i<500 or 1000 <= i < 1100 or 1600 <= i < 1700 or 2200 <= i < 2300 or 2800 <= i < 2900 or 3400 <= i < 3500:
                save_name_v = "target000067"
            else:
                save_name_v = "target000068"
            original_image = Image.open(image)
            dirname_v = '../target/' + "image_train_opp/" + date + "/"
            # save_name = os.path.join(dirname, save_name)
            save_name_t = os.path.join(dirname_v, save_name_v + '/' + image.split(os.path.sep)[-1])
            if not os.path.exists(os.path.dirname(save_name_t)):
                os.makedirs(os.path.dirname(save_name_t))
            # np.savetxt(save_name, behavior[i], fmt="%.6f")
            original_image.save(save_name_t)
            if i % 2 == 0:
                t = int(save_name_v.split('target')[-1])
                save_name_v = "target" + str(t + 72).zfill(6) + '/' + image.split(os.path.sep)[-1]
                dirname_v = '../target/' + "image_train_opp/" + date + "/"
                #save_name = os.path.join(dirname, save_name)
                save_name_t = os.path.join(dirname_v, save_name_v)
                if not os.path.exists(os.path.dirname(save_name_t)):
                    os.makedirs(os.path.dirname(save_name_t))
                #np.savetxt(save_name, behavior[i], fmt="%.6f")
                original_image.save(save_name_t)
        count = count+1
    return all_file_list

# Read image and turn it into 120x160
def read_input_folder(images_path, date, test):
    all_file_list = []
    dir_list = []
    for (root, dirs, files) in os.walk(images_path):
        dir_list.append(root)
        file_list = []
        for file in files:
            file_list.append(os.path.join(root, file))
        file_list.sort()
        if(len(file_list)>0):
            all_file_list.append(file_list)

    all_file_list.sort()
    max_len = 1
    count = 0
    #all_resized_images = np.zeros((len(all_file_list), max_len, 120, 160, 1))
    for file_list in all_file_list:
        num_of_images = len(file_list)
        file_list.sort()
        resized_images = []
        for i, image in enumerate(file_list):
            #if i % 2 == 0:
            original_image = Image.open(image)
            t = int(image.split(os.path.sep)[-2].split('t')[-1])
            save_name_v = "target" + str(t).zfill(6) + '/' + image.split(os.path.sep)[-1]#str(t + 72).zfill(6) + '/' + image.split(os.path.sep)[-1]
            if t in test: #(t + 72) in test:
                dirname_v = '../target/' + "image_test/" + date + "/"
            else:
                dirname_v = '../target/' + "image_train/" + date + "/"
                #save_name = os.path.join(dirname, save_name)
            save_name_v = os.path.join(dirname_v, save_name_v)
            if not os.path.exists(os.path.dirname(save_name_v)):
                os.makedirs(os.path.dirname(save_name_v))
                #np.savetxt(save_name, behavior[i], fmt="%.6f")
            original_image.save(save_name_v)
        count = count+1
    return all_file_list

def savefastimages():

    # get the training configuration (batch size, initialisation, num_of_iterations number, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    dates = ["201207", "201223", "210107", "210115", "210129", "210203"]
    test = [2, 6, 9, 14, 17, 25, 31, 33, 34, 38, 42, 45, 50, 53, 61, 67, 69, 70,
            74, 76, 77, 78, 79, 80, 81, 84, 94, 110, 112, 113, 114, 115, 116, 117, 120, 130]
    for date in dates:
        #B_data_dir = "../target/behavior/" + date
        V_data_dir = "../target/image_train_t/" + date


        # get the joint angles for actions
        #B_fw, B_bw, B_bin, B_len, filenames = read_sequential_target(B_data_dir, True)
        #B_fw_fast = B_fw[0::2, :, :]
        #B_bin_fast = B_bin[0::2, :, :]
        # get the visual features for action images
        V_fw = read_input_folder(V_data_dir, date, test)

if __name__ == "__main__":
    organise_images_three_cubes("../../nico_moves_cubes/images")
    organise_joint_angles("../../nico_moves_cubes/logs")
    #organise_images_train_test("../target/image-train")
    #savefastimages()
