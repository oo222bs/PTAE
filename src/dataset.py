import string

from torch.utils.data import Dataset
from torchvision import transforms
from data_util import read_sequential_target, read_sequential_target_lang#, read_sequential_target_pickle
import os
import json
import xml.etree.cElementTree as ET
import logging
from PIL import Image
import numpy as np
import torch

class RLBenchMTLDataset(Dataset):
    def __init__(self, dataset_dirs, split='train', max_len=None, pose=False, key_out=False):
        self.key_out = key_out
        # get the dataset folders
        if split == 'test':
            lang_dir = dataset_dirs.L_dir_test
            lang_oh_dir = dataset_dirs.L_oh_dir_test
            joints_dir = dataset_dirs.B_dir_test
            pose_dir = dataset_dirs.P_dir_test
            if self.key_out:
                pose_out_dir = dataset_dirs.P_out_dir_test
        elif split == "val":
            lang_dir = dataset_dirs.L_dir_val
            lang_oh_dir = dataset_dirs.L_oh_dir_val
            joints_dir = dataset_dirs.B_dir_val
            pose_dir = dataset_dirs.P_dir_val
            if self.key_out:
                pose_out_dir = dataset_dirs.P_out_dir_val
        else:
            lang_dir = dataset_dirs.L_dir
            lang_oh_dir = dataset_dirs.L_oh_dir
            joints_dir = dataset_dirs.B_dir
            pose_dir = dataset_dirs.P_dir
            if self.key_out:
                pose_out_dir = dataset_dirs.P_out_dir

        # get the descriptions
        self.L_fw, self.L_filenames = read_sequential_target_lang(lang_dir, True)
        self.L_oh_fw, self.L_oh_bw, self.L_oh_bin, self.L_len, self.L_oh_filenames = read_sequential_target(lang_oh_dir, True)

        # get the joint angles or gripper poses for actions
        if pose:
            self.B_fw, self.B_bw, self.B_bin, self.B_len, self.B_filenames = read_sequential_target(pose_dir, True,
                                                                                                    max_len=max_len)
        else:
            self.B_fw, self.B_bw, self.B_bin, self.B_len, self.B_filenames = read_sequential_target(joints_dir, True, max_len=max_len)

        if self.key_out:
            self.B_out_fw, self.B_out_bw, self.B_out_bin, self.B_out_len, self.B_out_filenames = read_sequential_target(pose_out_dir, True,
                                                                                                    max_len=max_len-1)

        # before normalisation save max and min joint angles to variables (will be used when converting norm to original values)
        self.maximum_joint = self.B_fw.max()
        self.minimum_joint = self.B_fw.min()

        # get the visual features for action images
        #self.V_fw, self.V_bw, self.V_bin, self.V_len = read_sequential_target(vis_dir)
        #self.V_opp_fw, self.V_opp_bw, self.V_opp_bin, self.V_opp_len = read_sequential_target(vis_opp_dir)
        #self.V_fw, self.V_bw, self.V_bin, self.V_len = read_sequential_target_pickle(vis_dir)

        # create variables for data shapes
        #self.L_shape = (self.L_fw.shape[0] // 8, self.L_fw.shape[1], self.L_fw.shape[2])
        #self.L_shape = (self.L_fw.shape[0] // 4, self.L_fw.shape[1], self.L_fw.shape[2])
        self.L_shape = (self.L_oh_fw.shape[0], self.L_oh_fw.shape[1], self.L_oh_fw.shape[2])
        self.B_shape = self.B_fw.shape
        #self.V_shape = self.V_fw.shape

    def __len__(self):
        return len(self.L_len)

    def __getitem__(self, index):
        items = {}
        items["L_fw"] = self.L_fw[index]
        items["L_oh_fw"] = self.L_oh_fw[:, index, :]
        items["L_bw"] = self.L_oh_bw[:, index, :]
        items["B_fw"] = self.B_fw[:, index, :]
        items["B_bw"] = self.B_bw[:, index, :]
        items["B_out_fw"] = self.B_out_fw[:, index, :] if self.key_out else None
        items["B_out_bw"] = self.B_out_bw[:, index, :] if self.key_out else None
        items["L_len"] = self.L_len[index]  # 1 description per action
        items["B_len"] = self.B_len[index]
        items["B_bin"] = self.B_bin[:, index, :]
        items["B_out_bin"] = self.B_out_bin[:, index, :] if self.key_out else None
        items["L_filenames"] = self.L_filenames[index]
        items["L_oh_filenames"] = self.L_oh_filenames[index]
        items["B_filenames"] = self.B_filenames[index]
        items["B_out_filenames"] = self.B_out_filenames[index] if self.key_out else None
        items["max_joint"] = self.maximum_joint
        items["min_joint"] = self.minimum_joint
        return items

class PairedNico2BlocksDataset(Dataset):
    def __init__(self, dataset_dirs, test=False):
        # get the dataset folders
        if test:
            lang_dir = dataset_dirs.L_dir_test
            joints_dir = dataset_dirs.B_dir_test
            vis_dir = dataset_dirs.V_dir_test
            #vis_opp_dir = dataset_dirs.V_opp_dir_test
        else:
            lang_dir = dataset_dirs.L_dir
            joints_dir = dataset_dirs.B_dir
            vis_dir = dataset_dirs.V_dir
            #vis_opp_dir = dataset_dirs.V_opp_dir

        # get the descriptions
        self.L_fw, self.L_bw, self.L_bin, self.L_len, self.L_filenames = read_sequential_target(lang_dir, True)

        # get the joint angles for actions
        self.B_fw, self.B_bw, self.B_bin, self.B_len, self.B_filenames = read_sequential_target(joints_dir, True)

        # before normalisation save max and min joint angles to variables (will be used when converting norm to original values)
        self.maximum_joint = self.B_fw.max()
        self.minimum_joint = self.B_fw.min()

        # get the visual features for action images
        #self.V_fw, self.V_bw, self.V_bin, self.V_len = read_sequential_target(vis_dir)
        #self.V_opp_fw, self.V_opp_bw, self.V_opp_bin, self.V_opp_len = read_sequential_target(vis_opp_dir)
        #self.V_fw, self.V_bw, self.V_bin, self.V_len = read_sequential_target_pickle(vis_dir)

        # create variables for data shapes
        #self.L_shape = (self.L_fw.shape[0] // 8, self.L_fw.shape[1], self.L_fw.shape[2])
        self.L_shape = (self.L_fw.shape[0] // 4, self.L_fw.shape[1], self.L_fw.shape[2])
        #self.L_shape = (self.L_fw.shape[0], self.L_fw.shape[1], self.L_fw.shape[2])
        self.B_shape = self.B_fw.shape
        #self.V_shape = self.V_fw.shape

    def __len__(self):
        return len(self.L_len)

    def __getitem__(self, index):
        items = {}
        items["L_fw"] = self.L_fw[:, index, :]
        items["L_bw"] = self.L_bw[:, index, :]
        items["B_fw"] = self.B_fw[:, index, :]
        items["B_bw"] = self.B_bw[:, index, :]
        #items["V_fw"] = self.V_fw[:, index, :]
        #items["V_bw"] = self.V_bw[:, index, :]
        #items["V_opp_fw"] = self.V_opp_fw[:, index, :]
        #items["V_opp_bw"] = self.V_opp_bw[:, index, :]
        #items["L_len"] = self.L_len[index]  # 1 description per action
        items["L_len"] = self.L_len[index] / 4     # 4 alternatives per description
        #items["L_len"] = self.L_len[index] / 8     # 8 alternatives per description
        items["B_len"] = self.B_len[index]
        #items["V_len"] = self.V_len[index]
        #items["V_opp_len"] = self.V_opp_len[index]
        items["B_bin"] = self.B_bin[:, index, :]
        items["L_filenames"] = self.L_filenames[index]
        items["B_filenames"] = self.B_filenames[index]
        items["max_joint"] = self.maximum_joint
        items["min_joint"] = self.minimum_joint
        return items


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ImageDataset(Dataset):
    def __init__(self, im_dir, test=False):
        if test:
            phase = 'test'
        else:
            phase = 'train'
        #noise_flag = torch.randint(2, (1,))
        # Data augmentation and normalization for training
        # Just normalization for validation
        image_transforms = {
            'train': transforms.Compose([
                #transforms.RandomResizedCrop(224).to('cuda'),
                #transforms.RandomHorizontalFlip().to('cuda'),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to('cuda'),
                #transforms.RandomApply([transforms.RandomAffine(degrees=0, translate=(0.15,0.15), scale=(1.15, 1.15))], p=0.9)
                #affine(original_image, 0, [translation, 0], scale=1 + (translation / 160), shear=[0, 0])
                #transforms.RandomApply([AddGaussianNoise(0, 0.5)], p=noise_flag)
                #transforms.RandomApply([transforms.GaussianBlur(5)], p=noise_flag)
                #transforms.RandomChoice([AddGaussianNoise(0, 1), transforms.GaussianBlur(5)], p=0.5)
            ]),
            'test': transforms.Compose([
                #transforms.Resize(224).to('cuda'),
                #transforms.CenterCrop(224).to('cuda'),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).to('cuda')
            ]),
        }

        all_imgs = os.listdir(im_dir)
        all_imgs.sort(key=self.sort_by_int)
        self.length = len(all_imgs)
        transformed_images = []
        for img in all_imgs:
            original_image = Image.open(im_dir + '/' +img)
            transformed_image = image_transforms[phase](original_image).to('cuda')
            transformed_images.append(torch.unsqueeze(transformed_image, 0))
            original_image.close()

        self.transformed_images = torch.cat(transformed_images)

    def __len__(self):
        return len(self.length)

    def __getitem__(self, index):
        return self.transformed_images[index]

    def sort_by_int(self, e):
        return int(e.split('.')[0])

class KITMotionLanguageDataset(Dataset):
    def __init__(self, input_path):
        #from sklearn.preprocessing import OneHotEncoder
        #from sklearn.preprocessing import LabelEncoder
        import h5py
        f = h5py.File(input_path, 'r')
        #annotations_in = f['annotation_inputs']
        annotations_tar = f['annotation_targets']
        #motions_in = f['motion_inputs']
        motions_tar = f['motion_targets']
        mapping = f['mapping']
        id = f['ids']
        vocabulary = f['vocabulary']
        nb_vocabulary = vocabulary.shape[0]
        #self.vocabulary = []
        #for i in range(nb_vocabulary):
        #    self.vocabulary.append(vocabulary[i].decode())
        #start_symbol = f['vocabulary'].attrs['start_symbol']
        #start_idx = b'SOS'#list(vocabulary).index(start_symbol)
        nb_joints = len(f['motion_targets'].attrs['joint_names'])

        # Create usable data for training.
        #X_language = []
        #X_motion = []
        self.all_motions = []
        Y_language = []
        self.all_ids = []
        for motion_idx, annotation_idx, id_idx in mapping:
            #X_language.append(annotations_in[annotation_idx])
            #X_motion.append(motions_in[motion_idx])
            self.all_motions.append(motions_tar[motion_idx])
            Y_language.append(annotations_tar[annotation_idx])
            self.all_ids.append(id[id_idx])
        #assert len(X_language) == len(Y_motion)
        #assert len(X_motion) == len(Y_language)
        assert len(self.all_motions) == len(Y_language)
        #X_language = np.array(X_language)
        self.all_motions = np.array(self.all_motions)
        self.all_ids = np.array(self.all_ids)
        self.all_motions_bin = np.where(self.all_motions != 0.0, 1, self.all_motions)
        #X_motion = np.array(X_motion).astype('float32')
        Y_language = np.array(Y_language).astype('int32')
        self.L_len = Y_language.argmin(axis=1).T
        # Move language one back, since this is the previous time step.
        #X_language_dec = np.array(Y_language).astype('int32')[:, :-1]
        #X_language_dec = np.hstack([np.ones((X_language_dec.shape[0], 1), dtype='int32') * start_idx, X_language_dec])

        # Encode targets as probabilities.
        self.all_annotations = np.zeros((Y_language.shape[0], Y_language.shape[1], nb_vocabulary))
        for i in range(0,Y_language.shape[0]):
            for j in range(0, Y_language.shape[1]):
                self.all_annotations[i,j,Y_language[i,j]] = 1

    def __len__(self):
        return len(self.all_ids)

    def __getitem__(self, index):
        items = {}
        items["ids"] = self.all_ids[index]
        #items["vocabulary"] = self.vocabulary[index]
        items["B_fw"] = self.all_motions[index,:, :]
        items["B_bin"] = self.all_motions_bin[index, :, :]
        items["L_fw"] = self.all_annotations[index, :, :]
        items["L_len"] = self.L_len[index]
        #items["meta"] = self.all_metadata[index]
        return items
