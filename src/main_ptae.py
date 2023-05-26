import datetime
import math
import os

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from config import PtaeConfig, TrainConfig
from data_util import add_active_feature, normalise, pad_with_zeros
from dataset import PairedNico2BlocksDataset
from ptae import (
    PTAE,
    CrossmodalTransformerDecoderTransformer,
    train,
    validate,
)


def main():
    # get the network configuration (parameters such as number of layers and units)
    paramaters = PtaeConfig()
    paramaters.set_conf("../train/ptae_conf.txt")

    # get the training configuration
    # (batch size, initialisation, num_of_iterations number, saving and loading directory)
    train_conf = TrainConfig()
    train_conf.set_conf("../train/train_conf.txt")
    seed = train_conf.seed
    batch_size = train_conf.batch_size
    num_of_epochs = train_conf.num_of_epochs
    learning_rate = train_conf.learning_rate
    save_dir = train_conf.save_dir
    app_length = True
    if not os.path.exists(os.path.dirname(save_dir)):
        os.mkdir(os.path.dirname(save_dir))

    # Random Initialisation
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Use GPU if possible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    print(
        "The currently selected GPU is number:",
        torch.cuda.current_device(),
        ", it's a ",
        torch.cuda.get_device_name(device=None),
    )
    # Create a model instance
    model = PTAE(paramaters, app_length=app_length).to(device)
    # Initialise the optimiser
    # optimiser = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adam optimiser
    scheduler = MultiStepLR(optimiser, milestones=[10000], gamma=0.5)
    # scheduler = ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=8000)
    #  Inspect the model with tensorboard
    model_name = "ptae"
    date = str(datetime.datetime.now()).split(".")[0]
    writer = SummaryWriter(
        log_dir=".././logs/" + model_name + date
    )  # initialize the writer with folder "./logs"

    # Load the trained model
    # checkpoint = torch.load(save_dir + '/ptae.tar')       # get the checkpoint
    # model.load_state_dict(checkpoint['model_state_dict'])       # load the model state
    # optimiser.load_state_dict(checkpoint['optimiser_state_dict'])   # load the optimiser state

    model.train()  # tell the model that it's training time

    # Load the dataset
    training_data = PairedNico2BlocksDataset(train_conf)
    test_data = PairedNico2BlocksDataset(train_conf, True)

    # Get the max and min values for normalisation
    max_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).max()
    min_joint = np.concatenate((training_data.B_fw, test_data.B_fw), 1).min()
    max_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).max()
    min_vis = np.concatenate((training_data.V_fw, test_data.V_fw), 1).min()
    # max_vis_opp = np.concatenate((training_data.V_opp_fw, test_data.V_opp_fw), 1).max()
    # min_vis_opp = np.concatenate((training_data.V_opp_fw, test_data.V_opp_fw), 1).min()

    # normalise the joint angles, visual features between -1 and 1 and pad the fast actions with zeros
    training_data.B_bw = (
        normalise(training_data.B_bw, max_joint, min_joint) * training_data.B_bin
    )
    training_data.B_fw = (
        normalise(training_data.B_fw, max_joint, min_joint) * training_data.B_bin
    )
    test_data.B_bw = normalise(test_data.B_bw, max_joint, min_joint) * test_data.B_bin
    test_data.B_fw = normalise(test_data.B_fw, max_joint, min_joint) * test_data.B_bin
    training_data.V_bw = (
        normalise(training_data.V_bw, max_vis, min_vis) * training_data.V_bin
    )
    training_data.V_fw = (
        normalise(training_data.V_fw, max_vis, min_vis) * training_data.V_bin
    )
    test_data.V_bw = normalise(test_data.V_bw, max_vis, min_vis) * test_data.V_bin
    test_data.V_fw = normalise(test_data.V_fw, max_vis, min_vis) * test_data.V_bin
    # training_data.V_opp_bw = normalise(training_data.V_opp_bw, max_vis_opp, min_vis_opp) * training_data.V_opp_bin
    # training_data.V_opp_fw = normalise(training_data.V_opp_fw, max_vis_opp, min_vis_opp) * training_data.V_opp_bin
    # test_data.V_opp_bw = normalise(test_data.V_opp_bw, max_vis, min_vis_opp) * test_data.V_opp_bin
    # test_data.V_opp_fw = normalise(test_data.V_opp_fw, max_vis, min_vis_opp) * test_data.V_opp_bin
    # Add the binary active features as the last dimension to joints B
    if app_length == False:
        training_data.B_bw = add_active_feature(training_data.B_bw, training_data.B_bin)
        training_data.B_fw = add_active_feature(training_data.B_fw, training_data.B_bin)
        test_data.B_bw = add_active_feature(test_data.B_bw, test_data.B_bin)
        test_data.B_fw = add_active_feature(test_data.B_fw, test_data.B_bin)

    # Load the training and testing sets with DataLoader
    train_dataloader = DataLoader(
        training_data, batch_size=batch_size, shuffle=True
    )  # shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    step = 0
    # super_perc = 2
    # super_batches = np.random.choice(range(len(train_dataloader)), math.ceil(len(train_dataloader) * super_perc / 100), replace=False)

    # Training
    for epoch in range(num_of_epochs):
        epoch_loss = []
        epoch_loss_describe = []
        epoch_loss_execute = []
        # iter_within_epoch = 0
        for input in train_dataloader:
            input["L_fw"] = input["L_fw"].transpose(0, 1)
            input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
            input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
            input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
            input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
            # input["V_opp_fw"] = input["V_opp_fw"].transpose(0, 1).to(device)
            # input["V_opp_bw"] = input["V_opp_bw"].transpose(0, 1).to(device)
            input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
            input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
            # input["VB_fw"] = [input["V_fw"][0, :, :], input["B_fw"][0, :, :]]
             sentence_idx = 0#np.random.randint(8)  # Generate random index for description alternatives
            # Choose one of the eight description alternatives according to the generated random index
            L_fw_feed = input["L_fw"][4*sentence_idx:4+4*sentence_idx, :, :]#input["L_fw"]  # [5*sentence_idx:5+5*sentence_idx, :, :]#input["L_fw"]#[4*sentence_idx:4+4*sentence_idx, :, :]
            input["L_fw"] = L_fw_feed.to(device)
            # input["L_bw"] = L_bw_feed.to(device)
            # if iter_within_epoch in super_batches: #==5 or iter_within_epoch== 100
            #     supervised_sig = torch.randint(2,(1,))#3, (1,))
            #     if supervised_sig == 0:
            #         signal = 'describe'
            #     else:
            #         signal = 'execute'
            # else:
            #    rep_sig = torch.randint(2, (1,))
            #    if rep_sig == 0:
            #        signal = 'repeat action'
            #    else:
            #        signal = 'repeat language'
            # Train and print the losses
            l, b, t, signal = train(
                model, input, optimiser, epoch_loss, paramaters, vis_out=False
            )  # train_limited_data(model, input, optimiser, epoch_loss, paramaters, signal, vis_out=False)
            print(
                "step:{} total:{}, language:{}, behavior:{}, signal:{}".format(
                    step, t, l, b, signal
                )
            )
            step = step + 1
            if signal == "describe":
                epoch_loss_describe.append(epoch_loss[-1])
            elif signal == "execute":
                epoch_loss_execute.append(epoch_loss[-1])
            # iter_within_epoch = iter_within_epoch + 1

        writer.add_scalar(
            "Training Loss", np.mean(epoch_loss), epoch
        )  # add the overall loss to the Tensorboard
        writer.add_scalar(
            "Training Loss - Describe", np.mean(epoch_loss_describe), epoch
        )
        writer.add_scalar("Training Loss - Execute", np.mean(epoch_loss_execute), epoch)
        scheduler.step()
        # scheduler.step(np.mean(epoch_loss))
        # Testing
        if train_conf.test and (epoch + 1) % train_conf.test_interval == 0:
            epoch_loss_t = []
            epoch_loss_t_describe = []
            epoch_loss_t_execute = []
            for input in test_dataloader:
                input["L_fw"] = input["L_fw"].transpose(0, 1)
                input["B_fw"] = input["B_fw"].transpose(0, 1).to(device)
                input["V_fw"] = input["V_fw"].transpose(0, 1).to(device)
                input["B_bin"] = input["B_bin"].transpose(0, 1).to(device)
                # input["V_opp_fw"] = input["V_opp_fw"].transpose(0, 1).to(device)
                # input["V_opp_bw"] = input["V_opp_bw"].transpose(0, 1).to(device)
                input["VB_fw"] = [input["V_fw"][:, :, :], input["B_fw"][0, :, :]]
                # input["VB_fw"] = [input["V_fw"][0, :, :], input["B_fw"][0, :, :]]
                input["B_bw"] = input["B_bw"].transpose(0, 1).to(device)
                input["V_bw"] = input["V_bw"].transpose(0, 1).to(device)
                L_fw_feed = input["L_fw"][0:4, :, :]
                #sentence_idx = np.random.randint(8)  # Generate random index for description alternatives
                # Choose one of the eight description alternatives according to the generated random index
                #L_fw_feed = input["L_fw"]  # [5 * sentence_idx:5 + 5 * sentence_idx, :,]  # input["L_fw"]#[4*sentence_idx:4+4*sentence_idx, :, :]
                input["L_fw"] = L_fw_feed.to(device)

                # Calculate and print the losses
                l, b, t, signal = validate(
                    model, input, epoch_loss_t, paramaters, vis_out=False
                )
                print("test")
                print(
                    "step:{} total:{}, language:{}, behavior:{}, signal:{}".format(
                        step, t, l, b, signal
                    )
                )
                if signal == "describe":
                    epoch_loss_t_describe.append(epoch_loss_t[-1])
                elif signal == "execute":
                    epoch_loss_t_execute.append(epoch_loss_t[-1])
            writer.add_scalar(
                "Test Loss", np.mean(epoch_loss_t), epoch
            )  # add the overall loss to the Tensorboard
            writer.add_scalar(
                "Test Loss - Describe", np.mean(epoch_loss_t_describe), epoch
            )
            writer.add_scalar(
                "Test Loss - Execute", np.mean(epoch_loss_t_execute), epoch
            )
        # Save the model parameters at every log interval
        if (epoch + 1) % train_conf.log_interval == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimiser_state_dict": optimiser.state_dict(),
                },
                save_dir + "/ptae.tar",
            )
    # Flush and close the summary writer of Tensorboard
    writer.flush()
    writer.close()


if __name__ == "__main__":
    main()
