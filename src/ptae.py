import torch
from torch import nn
import math


def train(model, batch, optimiser, epoch_loss, params, vis_out = False, appriori_len = True):
    optimiser.zero_grad()  # free the optimiser from previous gradients
    gt_description = batch['L_fw'][1:]
    if vis_out:
        gt_action = torch.cat((batch['V_fw'][1:],batch['B_fw'][1:]), dim=-1)
    else:
        gt_action = batch['B_fw'][1:]
    ran_sig = torch.randint(3, (1,))   #torch.randint(4, (1,))
    if ran_sig == 0:
        rep_sig = torch.randint(2, (1,))   #torch.randint(3, (1,))
        if rep_sig == 0:
            signal = 'repeat action'
            if appriori_len:
                gt_description = gt_description[-1].unsqueeze(0)
            else:
                gt_description = torch.zeros((1, gt_description.shape[1], gt_description.shape[2])).cuda()
                gt_description[:, :, 1] = 1
        #elif rep_sig == 1:
        #    signal = 'repeat both'
        else:
            signal = 'repeat language'
            if vis_out:
                if appriori_len == True:
                    gt_action = torch.cat((batch['V_fw'][0].repeat(len(gt_action), 1, 1) *
                                           batch["B_bin"][1:].repeat(1, 1, int(batch["V_fw"].shape[-1] / batch["B_bin"].shape[-1])),
                                           batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]), dim=-1)
                else:
                    gt_action = torch.cat((batch['V_fw'][0].repeat(len(gt_action), 1, 1) *
                                           batch["B_bin"][1:].repeat(1, 1, int(batch["V_fw"].shape[-1] / batch["B_bin"].shape[-1])),
                                           torch.cat((batch['B_fw'][0].repeat(len(gt_action), 1, 1)[:, :, :-1] * batch["B_bin"][1:],
                                                      batch['B_fw'][1: ,:, -1].unsqueeze(-1)), -1)), dim=-1)
            else:
                if appriori_len == True:
                    gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
                else:
                    gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
                    #gt_action = torch.cat((batch['B_fw'][0].repeat(len(gt_action), 1, 1)[:, :, :-1] * batch["B_bin"][1:],
                    #           batch['B_fw'][1:,:, -1].unsqueeze(-1)), -1)
    elif ran_sig == 1:
        signal = 'describe'
        if vis_out:
            if appriori_len == True:
                gt_action = torch.cat((batch['V_bw'][0].repeat(len(gt_action), 1, 1) *
                                        batch["B_bin"][1:].repeat(1, 1, int(batch["V_bw"].shape[-1] / batch["B_bin"].shape[-1])),
                                        batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]), dim=-1)
            else:
                gt_action = torch.cat((batch['V_bw'][0].repeat(len(gt_action), 1, 1) *
                                        batch["B_bin"][1:].repeat(1, 1, int(batch["V_bw"].shape[-1] / batch["B_bin"].shape[-1])),
                                       torch.cat((batch['B_bw'][0].repeat(len(gt_action), 1, 1)[:, :, :-1] * batch["B_bin"][1:],
                                                  batch['B_bw'][1:, :, -1].unsqueeze(-1)), -1)), dim=-1)
        else:
            if appriori_len == True:
                gt_action = batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
            else:
                gt_action = batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
                #gt_action = torch.cat((batch['B_bw'][0].repeat(len(gt_action), 1, 1)[:, :, :-1] * batch["B_bin"][1:],
                #                       batch['B_bw'][1:, :, -1].unsqueeze(-1)), -1)
    else:
        signal = 'execute'
        if appriori_len:
            gt_description = gt_description[-1].unsqueeze(0)
        else:
            gt_description = torch.zeros((1, gt_description.shape[1], gt_description.shape[2])).cuda()
            gt_description[:, :, 1] = 1

    output = model(batch, signal, appriori_len)
    L_loss, B_loss, batch_loss = loss(output, gt_description, gt_action, batch["B_bin"], signal, params, vis_out, appriori_len)  # compute loss
    batch_loss.backward()  # compute gradients
    optimiser.step()  # update weights
    epoch_loss.append(batch_loss.item()) # record the batch loss
    #scheduler.step()

    return L_loss, B_loss, batch_loss, signal  # return the losses

def validate(model, batch, epoch_loss, params, vis_out=False, appriori_len = True):
    with torch.no_grad():
        gt_description = batch['L_fw'][1:]
        if vis_out:
            gt_action = torch.cat((batch['V_fw'][1:], batch['B_fw'][1:]), dim=-1)
        else:
            gt_action = batch['B_fw'][1:]
        ran_sig = torch.randint(3, (1,))   #torch.randint(4, (1,))
        if ran_sig == 0:
            rep_sig = torch.randint(2, (1,))  #torch.randint(3, (1,))
            if rep_sig == 0:
                signal = 'repeat action'
                if appriori_len:
                    gt_description = gt_description[-1].unsqueeze(0)
                else:
                    gt_description = torch.zeros((1, gt_description.shape[1], gt_description.shape[2])).cuda()
                    gt_description[:,:,1] = 1
            #elif rep_sig == 1:
            #    signal = 'repeat both'
            else:
                signal = 'repeat language'
                if vis_out:
                    if appriori_len == True:
                        gt_action = torch.cat((batch['V_fw'][0].repeat(len(gt_action), 1, 1) *
                                               batch["B_bin"][1:].repeat(1, 1, int(
                                                   batch["V_fw"].shape[-1] / batch["B_bin"].shape[-1])),
                                               batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]),
                                              dim=-1)
                    else:
                        gt_action = torch.cat((batch['V_fw'][0].repeat(len(gt_action), 1, 1) *
                                               batch["B_bin"][1:].repeat(1, 1, int(
                                                   batch["V_fw"].shape[-1] / batch["B_bin"].shape[-1])),
                                               torch.cat((batch['B_fw'][0].repeat(len(gt_action), 1, 1)[:, :, :-1] *
                                                          batch["B_bin"][1:],
                                                          batch['B_fw'][1:, :, -1].unsqueeze(-1)), -1)), dim=-1)
                else:
                    if appriori_len == True:
                        gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
                    else:
                        gt_action = batch['B_fw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
                        #gt_action = torch.cat(
                        #    (batch['B_fw'][0].repeat(len(gt_action), 1, 1)[:, :, :-1] * batch["B_bin"][1:],
                        #     batch['B_fw'][1:, :, -1].unsqueeze(-1)), -1)
        elif ran_sig == 1:
            signal = 'describe'
            if vis_out:
                if appriori_len == True:
                    gt_action = torch.cat((batch['V_bw'][0].repeat(len(gt_action), 1, 1) *
                                           batch["B_bin"][1:].repeat(1, 1, int(
                                               batch["V_bw"].shape[-1] / batch["B_bin"].shape[-1])),
                                           batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]), dim=-1)
                else:
                    gt_action = torch.cat((batch['V_bw'][0].repeat(len(gt_action), 1, 1) *
                                           batch["B_bin"][1:].repeat(1, 1, int(
                                               batch["V_bw"].shape[-1] / batch["B_bin"].shape[-1])),
                                           torch.cat((batch['B_bw'][0].repeat(len(gt_action), 1, 1)[:, :, :-1] * batch[
                                                                                                                     "B_bin"][
                                                                                                                 1:],
                                                      batch['B_bw'][1:, :, -1].unsqueeze(-1)), -1)), dim=-1)
            else:
                if appriori_len == True:
                    gt_action = batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
                else:
                    gt_action = batch['B_bw'][0].repeat(len(gt_action), 1, 1) * batch["B_bin"][1:]
                    #gt_action = torch.cat(
                    #    (batch['B_bw'][0].repeat(len(gt_action), 1, 1)[:, :, :-1] * batch["B_bin"][1:],
                    #     batch['B_bw'][1:, :, -1].unsqueeze(-1)), -1)
        else:
           signal = 'execute'
           if appriori_len:
               gt_description = gt_description[-1].unsqueeze(0)
           else:
               gt_description = torch.zeros((1, gt_description.shape[1], gt_description.shape[2])).cuda()
               gt_description[:, :, 1] = 1

        output = model(batch, signal, appriori_len)
        L_loss, B_loss, batch_loss = loss(output, gt_description, gt_action, batch["B_bin"], signal, params, vis_out, appriori_len)  # compute loss
        epoch_loss.append(batch_loss.item())  # record the batch loss

    return L_loss, B_loss, batch_loss, signal # return the losses


def loss(output, gt_description, gt_action, B_bin, signal, net_conf, vis_out=False, fix_seq_len=True):
    if signal == 'repeat both':
        [L_output, B_output] = output
        if fix_seq_len:
            L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output), 2))
            if vis_out:
                B_output = B_output * B_bin[1:].repeat(1, 1, int(B_output.shape[-1] / B_bin.shape[-1]))
            else:
                B_output = B_output * B_bin[1:]
            B_loss = torch.mean(torch.square(B_output - gt_action))  # action loss (MSE)
        else:
            EPS = 1e-12
            if gt_description.size()[0] <= L_output.size()[0]:
                L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output[:gt_description.size()[0], :, :]), 2))
            else:
                L_padding = torch.zeros(gt_description.shape[0]- L_output.shape[0], gt_description.shape[1], gt_description.shape[2]) + EPS
                L_padding[:,:,2] = 1
                L_output = torch.cat((L_output, L_padding.cuda()), 0)
                L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output), 2))
            if vis_out:
                B_active = B_output[:, :, -1].unsqueeze(-1)
                if gt_action.size()[0] <= B_output.size()[0]:
                    B_output = B_output[:gt_action.shape[0], :, :-1] * B_bin[1:gt_action.shape[0] + 1].repeat(1, 1, int(
                        B_output.shape[-1] - 1 / B_bin.shape[-1]))
                else:
                    B_output = B_output[:, :, :-1] * B_bin[1:B_output.shape[0] + 1].repeat(1, 1, int(
                        B_output.shape[-1] - 1 / B_bin.shape[-1]))
            else:
                B_active = B_output[:, :, -1].unsqueeze(-1)
            if gt_action.size()[0] <= B_output.size()[0]:
                B_output = B_output[:gt_action.shape[0], :, :-1] * B_bin[1:gt_action.shape[0] + 1]
                B_out_loss = torch.mean(torch.square(B_output - gt_action[:, :, :-1]))
                B_active_loss = torch.mean(-torch.sum(gt_action[:, :, -1].unsqueeze(-1) * torch.log(B_active[:gt_action.shape[0], :, :])
                                                      + (torch.ones((gt_action.shape[0], gt_action.shape[1], 1)).cuda() - gt_action[:, :,-1].unsqueeze(-1))
                                            * torch.log(torch.ones((gt_action.shape[0], gt_action.shape[1], 1)).cuda() - B_active[:gt_action.shape[0]]), 2))
                B_loss = B_out_loss + B_active_loss
            else:
                B_output = B_output[:, :, :-1] * B_bin[1:B_output.shape[0] + 1]
                B_active = torch.cat((B_active, torch.zeros(gt_action.shape[0] - B_active.shape[0], B_active.shape[1],
                                        B_active.shape[2]).cuda()), 0)
                B_output = torch.cat((B_output, torch.zeros(gt_action.shape[0] - B_output.shape[0], B_output.shape[1],
                                        B_output.shape[2]).cuda()), 0)
                B_out_loss = torch.mean(torch.square(B_output - gt_action[:, :, :-1]))
                B_active_loss = torch.mean(-torch.sum(gt_action[:, :, -1].unsqueeze(-1) * torch.log(B_active+EPS)
                                           +(torch.ones((gt_action.shape[0], gt_action.shape[1], 1)).cuda() - gt_action[:, :,-1].unsqueeze(-1))
                                            *torch.log(torch.ones((gt_action.shape[0], gt_action.shape[1], 1)).cuda() - B_active + EPS), 2))
                B_loss = B_out_loss + B_active_loss
    elif signal == 'describe' or signal == 'repeat language' or signal == 'describe action' or signal == 'describe color':
        [L_output, B_output] = output
        if fix_seq_len:
            L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output), 2))
            if vis_out:
                B_output = B_output * B_bin[1:].repeat(1, 1, int(B_output.shape[-1] / B_bin.shape[-1]))
            else:
                B_output = B_output * B_bin[1:]
            B_loss = torch.mean(torch.square(B_output - gt_action))  # action loss (MSE)
        else:
            EPS = 1e-12
            if gt_description.size()[0] <= L_output.size()[0]:
                L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output[:gt_description.size()[0], :, :]), 2))
            else:
                L_padding = torch.zeros(gt_description.shape[0] - L_output.shape[0], gt_description.shape[1], gt_description.shape[2]) + EPS
                L_padding[:,:,2] = 1
                L_output = torch.cat((L_output, L_padding.cuda()), 0)
                L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output), 2))
            if vis_out:
                B_active = B_output[:,:,-1].unsqueeze(-1)
                if gt_action.size()[0] <= B_output.size()[0]:
                    B_output = B_output[:gt_action.shape[0], :, :-1] * B_bin[1:gt_action.shape[0] + 1].repeat(1, 1, int(
                        B_output.shape[-1] - 1 / B_bin.shape[-1]))
                else:
                    B_output = B_output[:,:,:-1] * B_bin[1:B_output.shape[0]+1].repeat(1, 1, int(B_output.shape[-1]-1 / B_bin.shape[-1]))
            else:
                if net_conf.B_binary_dim == 1:
                    B_active = B_output[:, :, -net_conf.B_binary_dim:].unsqueeze(-1)
                    gt_active = gt_action[:, :, -1].unsqueeze(-1)
                else:
                    B_active = B_output[:, :, -net_conf.B_binary_dim:]
                    gt_active = gt_action[:, :, -net_conf.B_binary_dim:]
            if gt_action.size()[0] < B_output.size()[0]:
                B_output = B_output[:gt_action.shape[0], :, :-net_conf.B_binary_dim] * B_bin[1:gt_action.shape[0] + 1]
                B_out_loss = torch.mean(torch.square(B_output - gt_action[:, :, :-net_conf.B_binary_dim]))
                B_active_loss = torch.mean(-torch.sum(gt_active * torch.log(B_active[:gt_active.shape[0], :, :] + EPS)
                                           +(torch.ones((gt_active.shape[0], gt_active.shape[1], 1)).cuda() - gt_active)
                                            *torch.log(torch.ones((gt_active.shape[0], gt_active.shape[1], 1)).cuda() - B_active[:gt_action.shape[0]] + EPS), 2))
                B_loss = B_out_loss + B_active_loss
            else:
                B_output = B_output[:, :, :-net_conf.B_binary_dim] * B_bin[1:,:,:-net_conf.B_binary_dim]#B_bin[1:B_output.shape[0] + 1]
                B_active = torch.cat((B_active, torch.zeros(gt_action.shape[0] - B_active.shape[0], B_active.shape[1],
                                        B_active.shape[2]).cuda()), 0)
                B_output = torch.cat((B_output, torch.zeros(gt_action.shape[0] - B_output.shape[0], B_output.shape[1],
                                        B_output.shape[2]).cuda()), 0)
                B_out_loss = torch.mean(torch.square(B_output - gt_action[:, :, :-net_conf.B_binary_dim]))
                B_active_loss = torch.mean(-torch.sum(gt_active * torch.log(B_active + EPS)
                                           +(torch.ones((gt_active.shape[0], gt_active.shape[1], 1)).cuda() - gt_active)
                                            *torch.log(torch.ones((gt_active.shape[0], gt_active.shape[1], 1)).cuda() - B_active + EPS), 2))
                B_loss = B_out_loss + B_active_loss
    else:
        [L_output, B_output] = output
        if fix_seq_len:
            L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output), 2))
            if vis_out:
                B_output = B_output * B_bin[1:].repeat(1, 1, int(B_output.shape[-1] / B_bin.shape[-1]))
            else:
                B_output = B_output * B_bin[1:]
            B_loss = torch.mean(torch.square(B_output - gt_action))  # action loss (MSE)
        else:
            EPS = 1e-12
            if gt_description.size()[0] <= L_output.size()[0]:
                L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output[:gt_description.size()[0], :, :]), 2))
            else:
                L_padding = torch.zeros(gt_description.shape[0]- L_output.shape[0], gt_description.shape[1], gt_description.shape[2]) + EPS
                L_padding[:,:,2] = 1
                L_output = torch.cat((L_output, L_padding.cuda()), 0)
                L_loss = torch.mean(-torch.sum(gt_description * torch.log(L_output), 2))
            if vis_out:
                B_active = B_output[:,:,-1].unsqueeze(-1)
                if gt_action.size()[0] <= B_output.size()[0]:
                    B_output = B_output[:gt_action.shape[0], :, :-1] * B_bin[1:gt_action.shape[0] + 1].repeat(1, 1, int(
                        B_output.shape[-1] - 1 / B_bin.shape[-1]))
                else:
                    B_output = B_output[:,:,:-1] * B_bin[1:B_output.shape[0]+1].repeat(1, 1, int(B_output.shape[-1]-1 / B_bin.shape[-1]))
            else:
                if net_conf.B_binary_dim == 1:
                    B_active = B_output[:, :, -1].unsqueeze(-1)
                    gt_active = gt_action[:, :, -1].unsqueeze(-1)
                else:
                    B_active = B_output[:, :, -net_conf.B_binary_dim:]
                    gt_active = gt_action[:, :, -net_conf.B_binary_dim:]
            if gt_action.size()[0] < B_output.size()[0]:
                B_output = B_output[:gt_action.shape[0], :, :-net_conf.B_binary_dim] * B_bin[1:gt_action.shape[0] + 1]
                B_out_loss = torch.mean(torch.square(B_output - gt_action[:, :, :-net_conf.B_binary_dim]))
                B_active_loss = torch.mean(-torch.sum(gt_active * torch.log(B_active[:gt_active.shape[0], :, :] + EPS)
                                           +(torch.ones((gt_active.shape[0], gt_active.shape[1], 1)).cuda() - gt_active)
                                            *torch.log(torch.ones((gt_active.shape[0], gt_active.shape[1], 1)).cuda() - B_active[:gt_active.shape[0]] + EPS), 2))
                B_loss = B_out_loss + B_active_loss
            else:
                B_output = B_output[:,:,:-net_conf.B_binary_dim] * B_bin[1:,:,:-net_conf.B_binary_dim]#B_bin[1:B_output.shape[0]+1]
                B_active = torch.cat((B_active, torch.zeros(gt_action.shape[0] - B_active.shape[0], B_active.shape[1],
                                        B_active.shape[2]).cuda()), 0)
                B_output = torch.cat((B_output, torch.zeros(gt_action.shape[0] - B_output.shape[0], B_output.shape[1],
                                        B_output.shape[2]).cuda()), 0)
                B_out_loss = torch.mean(torch.square(B_output - gt_action[:, :, :-net_conf.B_binary_dim]))
                B_active_loss = torch.mean(-torch.sum(gt_active * torch.log(B_active + EPS)
                                           +(torch.ones((gt_active.shape[0], gt_active.shape[1], 1)).cuda() - gt_active)
                                            *torch.log(torch.ones((gt_active.shape[0], gt_active.shape[1], 1)).cuda() - B_active + EPS), 2))
                B_loss = B_out_loss + B_active_loss
    loss = net_conf.L_weight * L_loss + net_conf.B_weight * B_loss
    return L_loss, B_loss, loss

# Word Embedding Layer
class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim)
    def forward(self, x):
        return self.embed(x)

class PeepholeLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, peephole=False, forget_bias=0.0):
        super().__init__()
        self.input_sz = input_size
        self.hidden_size = hidden_size
        self.peephole = peephole
        self.W = nn.Parameter(torch.Tensor(input_size + hidden_size, hidden_size * 4))
        self.peep_i = nn.Parameter(torch.Tensor(hidden_size))
        self.peep_f = nn.Parameter(torch.Tensor(hidden_size))
        self.peep_o = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size * 4))
        self.forget_bias = forget_bias
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, x, sequence_len=None,
                init_states=None):
        """Assumes x is of shape (sequence, batch, feature)"""
        if sequence_len is None:
            seq_sz, bs, _ = x.size()
        else:
            seq_sz = sequence_len.max()
            _, bs, _ = x.size()
        hidden_seq = []
        if init_states is None:
            c_t, h_t = (torch.zeros(bs, self.hidden_size).to(x.device),
                        torch.zeros(bs, self.hidden_size).to(x.device))
        else:
            c_t, h_t = init_states

        HS = self.hidden_size
        for t in range(seq_sz):
            x_t = x[t, :, :]
            if sequence_len is not None:
                if sequence_len.min() <= t+1:
                    old_c_t = c_t.clone().detach()
                    old_h_t = h_t.clone().detach()
            # batch the computations into a single matrix multiplication
            lstm_mat = torch.cat([x_t, h_t], dim=1)
            if self.peephole:
                gates = lstm_mat @ self.W + self.bias
            else:
                gates = lstm_mat @ self.W + self.bias
                g_t = torch.tanh(gates[:, HS * 2:HS * 3])

            if self.peephole:
                i_t, j_t, f_t, o_t = (
                    (gates[:, :HS]),  # input
                    (gates[:, HS:HS * 2]),  # new input
                    (gates[:, HS * 2:HS * 3]),   # forget
                    (gates[:, HS * 3:])   # output
                )
            else:
                i_t, f_t, o_t = (
                    torch.sigmoid(gates[:, :HS]),  # input
                    torch.sigmoid(gates[:, HS:HS * 2]),# + self.forget_bias),  # forget
                    torch.sigmoid(gates[:, HS * 3:])  # output
                )

            if self.peephole:
                c_t = torch.sigmoid(f_t + self.forget_bias + c_t * self.peep_f) * c_t \
                      + torch.sigmoid(i_t + c_t * self.peep_i) * torch.tanh(j_t)
                h_t = torch.sigmoid(o_t + c_t * self.peep_o) * torch.tanh(c_t)
            else:
                c_t = f_t * c_t + i_t * g_t
                h_t = o_t * torch.tanh(c_t)

            out = h_t.clone()
            if sequence_len is not None:
                if sequence_len.min() <= t:
                    c_t = torch.where(torch.tensor(sequence_len).to(c_t.device) <= t, old_c_t.T, c_t.T).T
                    h_t = torch.where(torch.tensor(sequence_len).to(h_t.device) <= t, old_h_t.T, h_t.T).T
                    out = torch.where(torch.tensor(sequence_len).to(out.device) <= t, torch.zeros(out.shape).to(out.device).T, out.T).T

            hidden_seq.append(out.unsqueeze(0))
        hidden_seq = torch.cat(hidden_seq, dim=0)

        return hidden_seq, (h_t, c_t)

# Language Model-Based Language Encoder
class LanguageModel(nn.Module):
    def __init__(self, language_model='bert-base-uncased', one_hot=True):
        super(LanguageModel, self).__init__()
        self.language_model = language_model
        self.one_hot = one_hot
        if language_model == 'roberta':
            from transformers import RobertaTokenizer, RobertaModel
            self.tokeniser = RobertaTokenizer.from_pretrained('roberta-base')
            self.encoder = RobertaModel.from_pretrained('roberta-base')      # batch, seq, hidden
        elif language_model == 'distilbert':
            from transformers import DistilBertTokenizer, DistilBertModel
            self.tokeniser = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            self.encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')      # batch, seq, hidden
        elif language_model == 'albert-base':
            from transformers import AlbertTokenizer, AlbertModel
            self.tokeniser = AlbertTokenizer.from_pretrained('albert-base-v2')
            self.encoder = AlbertModel.from_pretrained('albert-base-v2')      # batch, seq, hidden
        elif language_model == 't5':
            from transformers import T5Tokenizer, T5Model
            self.tokeniser = T5Tokenizer.from_pretrained('t5-small')
            self.encoder = T5Model.from_pretrained('t5-small')      # batch, seq, hidden
        elif language_model == 'sentence-lm':
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')      # batch, seq, hidden
        elif language_model == 'sentence-t5':
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer.from_pretrained('sentence-transformers/sentence-t5-base')      # batch, seq, hidden
        elif language_model == 'clip':
            from transformers import CLIPTokenizer, CLIPTextModel
            self.tokeniser = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch16')
            self.encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch16')
        else:
            from transformers import BertTokenizer, BertModel
            self.tokeniser = BertTokenizer.from_pretrained('bert-base-uncased')
            self.encoder = BertModel.from_pretrained("bert-base-uncased")  # batch, seq, hidden
        # Uncomment below if no finetuning desired
        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, inp):
        descriptions = []
        # Use the vocabulary to feed descriptions
        if self.one_hot:
            file = open('../vocabulary.txt', 'r')
            vocab = file.read().splitlines()
            file.close()
            t = inp[:, :, :].argmax(axis=-1)
            for i in range(inp.shape[1]):
                sentence = ''
                for k in range(0, inp.shape[0]):
                    sentence += vocab[t[k, i]] + ' '
                descriptions.append(sentence)#descriptions.append(sentence.replace('<BOS/EOS>', '')[:-1])
        else:
            for sentence in inp:
                descriptions.append(sentence)
            # Different versions of the description:
            #descriptions.append(sentence.split(' ')[-2] + ' ' + sentence.split(' ')[0] + ' ' + sentence.split(' ')[1])
            #descriptions.append('could you ' + sentence[:-1] + '?')
            #descriptions.append(sentence.split(' ')[0] + ' ' + sentence.split(' ')[-2] + ' ' + sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4])
            #descriptions.append(sentence.split(' ')[-2] + ' ' + sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4] + ' ' + sentence.split(' ')[0])
            #descriptions.append(sentence.split(' ')[1] + ' ' + sentence.split(' ')[2] + ' ' + sentence.split(' ')[3] + ' ' + sentence.split(' ')[4] +' ' + sentence.split(' ')[5] + ' ' + sentence.split(' ')[0])
        if self.language_model=='sentence-lm' or self.language_model=='sentence-t5':
            v = self.encoder.encode(descriptions)
        else:
            encoded_input = self.tokeniser(descriptions, return_tensors='pt', padding=True)#self.tokeniser(descriptions, return_tensors='pt', padding='max_length', max_length=19)
            output = self.encoder(**encoded_input.to(self.encoder.device))
            v = self.mean_pooling(output, encoded_input['attention_mask']).unsqueeze(1)     # use this when descriptions have different number of subwords to ignore padding
            #v = output.last_hidden_state#.mean(1)
        return v

    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class ResNetFeatExtractor(nn.Module):
    def __init__(self, layers=34, max_seq_len=100):
        super(ResNetFeatExtractor, self).__init__()
        from torchvision import models
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context
        if layers == 34:
            res34 = models.resnet34(weights=True)
            self.resnet = torch.nn.Sequential(*(list(res34.children())[:-3]))
        else:
            res18 = models.resnet18(weights=True)
            self.resnet = torch.nn.Sequential(*(list(res18.children())[:-3]))
        for param in self.resnet.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(p=0.10)
        self.firstconv = nn.Conv2d(256, 128, 3, padding=1)
        self.elu = nn.ELU()
        self.secondconv = nn.Conv2d(128, 128, 3, padding=1)
        self.linear = nn.Linear(128*8*8, 512)#nn.Linear(128*8*10, 512)#nn.Linear(128*14*14, 512)
        self.linear_s = nn.Linear(512, 30)
        self.tanh = nn.Tanh()
        self.max_seq_len = max_seq_len

    def forward(self, inp, static=False, rtn_last=False):#(self, inp, backward=False):
        #self.resnet.eval()
        extracted_features = []
        if rtn_last:
            last_image_features = []
        for i in range(0, len(inp)):
            with torch.no_grad():
                resnet_features = self.resnet(inp[i])#'.squeeze()
            feats = self.dropout(resnet_features)
            feats = self.firstconv(feats)
            feats = self.elu(feats)
            feats = self.dropout(feats)
            feats = self.secondconv(feats)
            feats = self.elu(feats)
            feats = torch.flatten(feats, start_dim=1)
            feats = self.linear(feats)
            feats = self.linear_s(feats)
            extracted_feature = self.tanh(feats)
            if rtn_last:
                last_image_features.append(extracted_feature[-1].unsqueeze(0))
            if static == False:
                if len(extracted_feature) < self.max_seq_len:
                    extracted_feature = torch.nn.functional.pad(input=extracted_feature, pad=(0, 0, 0, self.max_seq_len-len(extracted_feature)), mode='constant',
                                               value=0)
            extracted_features.append(extracted_feature.unsqueeze(0))
        extracted_features = torch.cat(extracted_features)
        if rtn_last:
            last_image_features = torch.cat(last_image_features)
            return [extracted_features, last_image_features]
        return extracted_features

class EfficientNetFeatExtractor(nn.Module):
    def __init__(self, variant='b0', max_seq_len=100):
        super(EfficientNetFeatExtractor, self).__init__()
        from torchvision import models
        import ssl
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            # Legacy Python that doesn't verify HTTPS certificates by default
            pass
        else:
            # Handle target environment that doesn't support HTTPS verification
            ssl._create_default_https_context = _create_unverified_https_context

        if variant =='b0':
            efficientnet_b0 = models.efficientnet_b0(weights='EfficientNet_B0_Weights.IMAGENET1K_V1')
            self.efficientnet = torch.nn.Sequential(*(list(torch.nn.Sequential(*(list(efficientnet_b0.children())[:-2]))[0].children())[:-1]))
        elif variant == 'b1':
            efficientnet_b1 = models.efficientnet_b1(weights='EfficientNet_B1_Weights.IMAGENET1K_V1')
            self.efficientnet = torch.nn.Sequential(*(list(torch.nn.Sequential(*(list(efficientnet_b1.children())[:-2]))[0].children())[:-1]))
        elif variant == 'b2':
            efficientnet_b2 = models.efficientnet_b2(weights='EfficientNet_B2_Weights.IMAGENET1K_V1')
            self.efficientnet = torch.nn.Sequential(*(list(torch.nn.Sequential(*(list(efficientnet_b2.children())[:-2]))[0].children())[:-1]))
        elif variant == 'b3':
            efficientnet_b3 = models.efficientnet_b3(weights='EfficientNet_B3_Weights.IMAGENET1K_V1')
            self.efficientnet = torch.nn.Sequential(*(list(torch.nn.Sequential(*(list(efficientnet_b3.children())[:-2]))[0].children())[:-1]))
        elif variant == 'b4':
            efficientnet_b4 = models.efficientnet_b4(weights='EfficientNet_B4_Weights.IMAGENET1K_V1')
            self.efficientnet = torch.nn.Sequential(*(list(torch.nn.Sequential(*(list(efficientnet_b4.children())[:-2]))[0].children())[:-1]))
        elif variant == 'b5':
            efficientnet_b5 = models.efficientnet_b5(weights='EfficientNet_B5_Weights.IMAGENET1K_V1')
            self.efficientnet = torch.nn.Sequential(*(list(torch.nn.Sequential(*(list(efficientnet_b5.children())[:-2]))[0].children())[:-1]))
        elif variant == 'b6':
            efficientnet_b6 = models.efficientnet_b6(weights='EfficientNet_B6_Weights.IMAGENET1K_V1')
            self.efficientnet = torch.nn.Sequential(*(list(torch.nn.Sequential(*(list(efficientnet_b6.children())[:-2]))[0].children())[:-1]))
        else:
            efficientnet_b7 = models.efficientnet_b7(weights='EfficientNet_B7_Weights.IMAGENET1K_V1')
            self.efficientnet = torch.nn.Sequential(*(list(torch.nn.Sequential(*(list(efficientnet_b7.children())[:-2]))[0].children())[:-1]))

        for param in self.efficientnet.parameters():
            param.requires_grad = False
        self.dropout = nn.Dropout(p=0.10)
        self.firstconv = nn.Conv2d(320, 160, 3, padding=1)
        self.elu = nn.ELU()
        self.secondconv = nn.Conv2d(160, 160, 3, padding=1)
        self.linear = nn.Linear(160*4*5, 256)#nn.Linear(128*14*14, 512)
        self.linear_s = nn.Linear(256, 30)
        self.tanh = nn.Tanh()
        self.max_seq_len = max_seq_len

    def forward(self, inp, static=False):#(self, inp, backward=False):
        #self.resnet.eval()
        extracted_features = []
        for i in range(0, len(inp)):
            with torch.no_grad():
                effnet_features = self.efficientnet(inp[i])#'.squeeze()
            feats = self.dropout(effnet_features)
            feats = self.firstconv(feats)
            feats = self.elu(feats)
            feats = self.dropout(feats)
            feats = self.secondconv(feats)
            feats = self.elu(feats)
            feats = torch.flatten(feats, start_dim=1)
            feats = self.linear(feats)
            feats = self.linear_s(feats)
            extracted_feature = self.tanh(feats)
            if static == False:
                if len(extracted_feature) < self.max_seq_len:
                    extracted_feature = torch.nn.functional.pad(input=extracted_feature, pad=(0, 0, 0, self.max_seq_len-len(extracted_feature)), mode='constant',
                                               value=0)
            extracted_features.append(extracted_feature.unsqueeze(0))
        extracted_features = torch.cat(extracted_features)
        return extracted_features

class Encoder(nn.Module):
    def __init__(self, params, lang=False, lstm_type='peephole'):
        super(Encoder, self).__init__()
        self.params = params
        self.lstm_type = lstm_type
        if lang:
            self.enc_cells = torch.nn.Sequential()
            if self.lstm_type == 'bidirectional':
                self.enc_cells.add_module("ellstm", nn.LSTM(input_size=self.params.L_input_dim + 5,#self.params.L_num_units,#
                                                                     hidden_size=self.params.L_num_units,
                                                                     num_layers=self.params.L_num_layers,
                                                                     bidirectional=True))
            elif self.lstm_type == 'peephole':
                for i in range(self.params.L_num_layers):
                    if i == 0:
                        self.enc_cells.add_module("ellstm"+str(i), PeepholeLSTM(input_size=self.params.L_input_dim+5,
                                                                                    hidden_size=self.params.L_num_units,
                                                                                    peephole=True, forget_bias=0.8))
                        #self.enc_cells.add_module("ellstm"+str(i), PeepholeLSTM(input_size=50,
                                                                                    #hidden_size=self.params.L_num_units,
                                                                                    #peephole=True, forget_bias=0.8)
                    else:
                        self.enc_cells.add_module("ellstm"+str(i), PeepholeLSTM(input_size=self.params.L_num_units,
                                                                                    hidden_size=self.params.L_num_units,
                                                                                    peephole=True, forget_bias=0.8))
            else:
                self.enc_cells.add_module("ellstm", nn.LSTM(input_size=self.params.L_input_dim + 5,
                                                                     hidden_size=self.params.L_num_units,
                                                                     num_layers=self.params.L_num_layers,
                                                                     bidirectional=False))
        else:
            self.enc_cells = torch.nn.Sequential()
            if self.lstm_type == 'bidirectional':
                self.enc_cells.add_module("ealstm", nn.LSTM(input_size=self.params.VB_input_dim,
                                                                     hidden_size=self.params.VB_num_units,
                                                                     num_layers=self.params.VB_num_layers,
                                                                     bidirectional=True))
            elif self.lstm_type == 'peephole':
                for i in range(self.params.VB_num_layers):
                    if i == 0:
                        self.enc_cells.add_module("ealstm" + str(i), PeepholeLSTM(input_size=self.params.VB_input_dim,
                                                                                      hidden_size=self.params.VB_num_units,
                                                                                      peephole=True, forget_bias=0.8))
                    else:
                        self.enc_cells.add_module("ealstm" + str(i), PeepholeLSTM(input_size=self.params.VB_num_units,
                                                                                      hidden_size=self.params.VB_num_units,
                                                                                      peephole=True, forget_bias=0.8))
            else:
                self.enc_cells.add_module("ealstm", nn.LSTM(input_size=self.params.VB_input_dim,
                                                                     hidden_size=self.params.VB_num_units,
                                                                     num_layers=self.params.VB_num_layers,
                                                                     bidirectional=False))
    def forward(self, inp, sequence_length, lang=False):
        if lang:
            num_of_layers = self.params.L_num_layers
        else:
            num_of_layers = self.params.VB_num_layers
        layer_input = inp
        #states = []
        if self.lstm_type == 'peephole':
            for l in range(num_of_layers):
                enc_cell = self.enc_cells.__getitem__(l)
                hidden_seq, _ = enc_cell(layer_input.float().to('cuda'), sequence_len=sequence_length)
                layer_input = hidden_seq
        else:
            enc_cell = self.enc_cells.__getitem__(0)
            hidden_seq, _ = enc_cell(layer_input.float().to('cuda'))
            #    states.append((hn, cn))
        #states = tuple(map(torch.stack, zip(*states)))
        #final_state = torch.stack(states, dim=1)    # n_layers, 2, batch_size, n_units
        #final_state = final_state.permute(2,0,1,3)  # transpose to batchsize, n_layers, 2, n_units
        #final_state = torch.reshape(final_state, (int(final_state.shape[0]), -1))
        return hidden_seq.permute(1,0,2)#final_state#

class Decoder(nn.Module):
    def __init__(self, params, lang=False, vis_out=False, lstm_type='peephole', appriori_length=True):
        super(Decoder, self).__init__()
        self.params = params
        self.vis_out = vis_out
        self.lstm_type = lstm_type
        if lang:
            self.dec_cells = torch.nn.Sequential()
            for i in range(self.params.L_num_layers):
                if i == 0:
                    if self.lstm_type == 'peephole':
                        self.dec_cells.add_module("dllstm"+str(i), PeepholeLSTM(self.params.L_input_dim,
                                                                                self.params.L_num_units, True, forget_bias=0.8).to('cuda'))
                    else:
                        self.dec_cells.add_module("dllstm"+str(i), nn.LSTM(self.params.L_input_dim, self.params.L_num_units).to('cuda'))
                else:
                    if self.lstm_type == 'peephole':
                        self.dec_cells.add_module("dllstm"+str(i), PeepholeLSTM(self.params.L_num_units,
                                                                                self.params.L_num_units, True, forget_bias=0.8).to('cuda'))
                    else:
                        self.dec_cells.add_module("dllstm"+str(i), nn.LSTM(self.params.L_num_units, self.params.L_num_units).to('cuda'))
            self.linear = nn.Linear(self.params.L_num_units, self.params.L_input_dim)
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.dec_cells = torch.nn.Sequential()
            for i in range(self.params.VB_num_layers):
                if i == 0:
                    if self.lstm_type == 'peephole':
                        self.dec_cells.add_module("dalstm"+str(i), PeepholeLSTM(input_size=self.params.VB_input_dim,
                                                                                hidden_size=self.params.VB_num_units,
                                                                                peephole=True, forget_bias=0.8).to('cuda'))
                    else:
                        self.dec_cells.add_module("dalstm"+str(i), nn.LSTM(input_size=self.params.VB_input_dim,
                                                                                hidden_size=self.params.VB_num_units).to('cuda'))
                else:
                    if self.lstm_type == 'peephole':
                        self.dec_cells.add_module("dalstm"+str(i), PeepholeLSTM(input_size=self.params.VB_num_units,
                                                                                hidden_size=self.params.VB_num_units,
                                                                                peephole=True, forget_bias=0.8).to('cuda'))
                    else:
                        self.dec_cells.add_module("dalstm"+str(i), nn.LSTM(input_size=self.params.VB_num_units,
                                                                                hidden_size=self.params.VB_num_units).to('cuda'))
            if appriori_length:
                if vis_out:
                    self.linear = nn.Linear(self.params.VB_num_units, self.params.VB_input_dim)
                else:
                    self.linear = nn.Linear(self.params.VB_num_units, self.params.B_input_dim)
            else:
                self.sigmoid = nn.Sigmoid()
                if vis_out:
                    self.linear_pred = nn.Linear(self.params.VB_num_units, self.params.VB_input_dim-1)
                    self.linear_finish = nn.Linear(self.params.VB_num_units, 1)
                else:
                    self.linear_pred = nn.Linear(self.params.VB_num_units, self.params.B_input_dim-1)
                    self.linear_finish = nn.Linear(self.params.VB_num_units, 1)
            self.tanh = nn.Tanh()

    def forward(self, input, length=None, initial_state=None, lang=False, teacher_forcing=True):
        y = []

        if lang:
            initial_state = initial_state.view(initial_state.size()[0], self.params.L_num_layers, 2, self.params.L_num_units)
            initial_state = initial_state.permute(1, 2, 0, 3)
            if length == None:
                i=0
                file = open('../vocabularyRLpad.txt', 'r')
                vocab = file.read().splitlines()
                done = torch.zeros(input.shape[0], dtype=bool).cuda()
                while i<self.params.L_max_length-1:
                    dec_states = []
                    layer_input = input.unsqueeze(0)
                    if i == 0:
                        for j in range(self.params.L_num_layers):
                            dec_cell = self.dec_cells.__getitem__(j)
                            dec_state = (initial_state[j][0].float().to('cuda'), initial_state[j][1].float().to('cuda'))
                            if self.lstm_type == 'peephole':
                                output, (hx, cx) = dec_cell(layer_input.float().to('cuda'), init_states=dec_state)
                            else:
                                output, (hx, cx) = dec_cell(layer_input.float().to('cuda'),
                                                            (dec_state[0].unsqueeze(0).contiguous(),
                                                             dec_state[1].unsqueeze(0).contiguous()))
                            dec_state = (hx, cx)
                            dec_states.append(dec_state)
                            layer_input = output
                    else:
                        layer_input = out
                        for j in range(self.params.L_num_layers):
                            dec_cell = self.dec_cells.__getitem__(j)
                            dec_state = prev_dec_states[j]
                            if self.lstm_type == 'peephole':
                                output, (hx, cx) = dec_cell(layer_input, init_states=dec_state)
                            else:
                                output, (hx, cx) = dec_cell(layer_input.to('cuda'), dec_state)
                            dec_state = (hx, cx)
                            dec_states.append(dec_state)
                            layer_input = output
                    prev_dec_states = dec_states
                    linear = self.linear(layer_input)
                    out = self.softmax(linear)
                    y.append(out.squeeze())
                    for batch_ind in (done == False).nonzero(as_tuple=True)[0]:
                        if vocab[out.argmax(-1)[0][batch_ind]] == '<EOS>':
                            done[batch_ind] = True
                    if torch.all(done):
                        break
                    i = i+1
            else:
                for i in range(length - 1):
                    dec_states = []
                    layer_input = input.unsqueeze(0)
                    if i == 0:
                        for j in range(self.params.L_num_layers):
                            dec_cell = self.dec_cells.__getitem__(j)
                            dec_state = (initial_state[j][0].float().to('cuda'), initial_state[j][1].float().to('cuda'))
                            if self.lstm_type == 'peephole':
                                output, (hx, cx) = dec_cell(layer_input.float().to('cuda'), init_states=dec_state)
                            else:
                                output, (hx, cx) = dec_cell(layer_input.float().to('cuda'), (dec_state[0].unsqueeze(0).contiguous(),
                                                                                             dec_state[1].unsqueeze(0).contiguous()))
                            dec_state = (hx, cx)
                            dec_states.append(dec_state)
                            layer_input = output
                    else:
                        layer_input = out
                        for j in range(self.params.L_num_layers):
                            dec_cell = self.dec_cells.__getitem__(j)
                            dec_state = prev_dec_states[j]
                            if self.lstm_type == 'peephole':
                                output, (hx, cx) = dec_cell(layer_input, init_states=dec_state)
                            else:
                                output, (hx, cx) = dec_cell(layer_input.to('cuda'), dec_state)
                            dec_state = (hx, cx)
                            dec_states.append(dec_state)
                            layer_input = output
                    prev_dec_states = dec_states
                    linear = self.linear(layer_input)
                    out = self.softmax(linear)
                    y.append(out.squeeze())
        else:
            initial_state = initial_state.view(initial_state.size()[0], self.params.VB_num_layers, 2, self.params.VB_num_units)
            initial_state = initial_state.permute(1, 2, 0, 3)
            if length == None:
                done = torch.zeros(input[1].shape[0], dtype=bool).cuda()
                for i in range(self.params.B_max_length - 1):
                    if self.vis_out == False or teacher_forcing == True:
                        if len(input[0]) > i:
                            current_V_in = input[0][i]
                        else:
                            current_V_in = torch.zeros((input[0].shape[1], input[0].shape[2])).cuda()#input[0][len(input[0])-1]
                    dec_states = []
                    if i == 0:
                        if self.vis_out and teacher_forcing == False:
                            current_V_in = input[0]
                        current_B_in = input[-1]#input
                        layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)#current_B_in.unsqueeze(0)#
                        for j in range(self.params.VB_num_layers):
                            dec_state = (initial_state[j][0].float(), initial_state[j][1].float())
                            dec_cell = self.dec_cells.__getitem__(j)
                            if self.lstm_type == 'peephole':
                                output, (hx, cx) = dec_cell(layer_input.float(), init_states=dec_state)
                            else:
                                output, (hx, cx) = dec_cell(layer_input.float().to('cuda'), (dec_state[0].unsqueeze(0).contiguous(),
                                                                                             dec_state[1].unsqueeze(0).contiguous()))
                            dec_state = (hx, cx)
                            dec_states.append(dec_state)
                            layer_input = output
                    else:
                        if self.vis_out and teacher_forcing == False:
                            layer_input = out
                        else:
                            if self.vis_out and teacher_forcing == True:
                                current_B_in = out[:,:,30:].squeeze(dim=0)
                            else:
                                current_B_in = out.squeeze(dim=0)
                            layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)#current_B_in.unsqueeze(0)#
                        for j in range(self.params.VB_num_layers):
                            dec_cell = self.dec_cells.__getitem__(j)
                            dec_state = prev_dec_states[j]
                            if self.lstm_type == 'peephole':
                                output, (hx,cx) = dec_cell(layer_input.float(), init_states=dec_state)
                            else:
                                output, (hx, cx) = dec_cell(layer_input.float(), dec_state)
                            dec_state = (hx, cx)
                            dec_states.append(dec_state)
                            layer_input = output
                    prev_dec_states = dec_states
                    linear = self.linear_pred(layer_input)
                    out_dim = self.tanh(linear)
                    finish = self.linear_finish(layer_input)
                    out_finish = self.sigmoid(finish)
                    out = torch.cat((out_dim, out_finish), -1)#torch.cat((out_dim.squeeze(0), out_finish.squeeze(0)), -1)
                    y.append(out.squeeze(0))  #y.append(out)
                    for batch_ind in (done == False).nonzero(as_tuple=True)[0]:
                        if out_finish.squeeze(0)[batch_ind] < 0.5:
                            done[batch_ind] = True
                    if torch.all(done):
                        break
            else:
                for i in range(length - 1):
                    if self.vis_out == False or teacher_forcing == True:
                        current_V_in = input[0][i]
                    dec_states = []
                    if i == 0:
                        if self.vis_out and teacher_forcing == False:
                            current_V_in = input[0]
                        current_B_in = input[-1]#input
                        layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)#current_B_in.unsqueeze(0)#
                        for j in range(self.params.VB_num_layers):
                            dec_state = (initial_state[j][0].float(), initial_state[j][1].float())
                            dec_cell = self.dec_cells.__getitem__(j)
                            if self.lstm_type == 'peephole':
                                output, (hx, cx) = dec_cell(layer_input.float(), init_states=dec_state)
                            else:
                                output, (hx, cx) = dec_cell(layer_input.float().to('cuda'), (dec_state[0].unsqueeze(0).contiguous(),
                                                                                             dec_state[1].unsqueeze(0).contiguous()))
                            dec_state = (hx, cx)
                            dec_states.append(dec_state)
                            layer_input = output
                    else:
                        if self.vis_out and teacher_forcing == False:
                            layer_input = out
                        else:
                            if self.vis_out and teacher_forcing == True:
                                current_B_in = out[:,:,30:].squeeze(dim=0)
                            else:
                                current_B_in = out.squeeze(dim=0)
                            layer_input = torch.cat([current_V_in, current_B_in], dim=1).unsqueeze(0)#current_B_in.unsqueeze(0)#
                        for j in range(self.params.VB_num_layers):
                            dec_cell = self.dec_cells.__getitem__(j)
                            dec_state = prev_dec_states[j]
                            if self.lstm_type == 'peephole':
                                output, (hx,cx) = dec_cell(layer_input.float(), init_states=dec_state)
                            else:
                                output, (hx, cx) = dec_cell(layer_input.float(), dec_state)
                            dec_state = (hx, cx)
                            dec_states.append(dec_state)
                            layer_input = output
                    prev_dec_states = dec_states
                    linear = self.linear(layer_input)
                    out = self.tanh(linear)
                    y.append(out.squeeze())
        y = torch.stack(y, dim=0)
        return y


class PTAE(nn.Module):
    def __init__(self, params, lang_enc_type='LSTM', act_enc_type='LSTM', app_length=True):
        super(PTAE, self).__init__()
        from crossmodal_transformer import Visual_Ling_Attn as CMTransformer
        self.params = params
        self.lang_enc_type = lang_enc_type
        self.act_enc_type = act_enc_type

        if self.lang_enc_type == 'LSTM':
            self.lang_encoder = Encoder(self.params, True, lstm_type='bidirectional')
        elif self.lang_enc_type == 'BERT':
            self.lang_encoder = LanguageModel(self.params, 'bert-base')
        elif self.lang_enc_type == 'WordEmbedding':
            self.word_embedder = Embedder(self.params.L_input_dim + 5, emb_dim=params.L_num_units)

        if self.act_enc_type == 'LSTM':
            self.action_encoder = Encoder(self.params, False, lstm_type='bidirectional')

        self.hidden = CMTransformer(self.params)

        self.initial_lang = nn.Linear(self.params.hidden_dim, self.params.L_num_units*self.params.L_num_layers*2)
        self.initial_act = nn.Linear(self.params.hidden_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        self.lang_decoder = Decoder(self.params, True, lstm_type='regular', appriori_length=app_length)
        self.action_decoder = Decoder(self.params, False, lstm_type='regular', appriori_length=app_length)

    def forward(self, inp, signal, appriori_len=True):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim+4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim+2] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim+1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input=torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) *\
                     inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        if self.lang_enc_type == 'LSTM':
            encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        elif self.lang_enc_type == 'BERT':
            encoded_lang = self.lang_encoder(lang_inp)
        elif self.lang_enc_type == 'WordEmbedding':
            encoded_lang = self.word_embedder(lang_inp.argmax(axis=-1)).permute(1,0,2)
        else:
            encoded_lang = lang_inp.permute(1,0,2).float()

        if self.act_enc_type == 'LSTM':
            encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        else:
            encoded_act = VB_input.permute(1,0,2).float()#encoded_act_orig = VB_input.permute(1,0,2).float()#
            #encoded_act = []
            #ts_interval = int(encoded_act_orig.shape[1] / encoded_lang.shape[1])
            #for ts in range(0, encoded_lang.shape[1]):
            #    encoded_act.append(encoded_act_orig[:, ts*ts_interval, :])
            #encoded_act = torch.stack(encoded_act, dim=1)

        h = self.hidden(encoded_lang, encoded_act, None, None)
        h = h.mean(1)

        L_dec_init_state = self.initial_lang(h)
        VB_dec_init_state = self.initial_act(h)

        if signal == 'repeat both':
            VB_input_f = inp['VB_fw']
            if appriori_len:
                L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
            else:
                L_output = self.lang_decoder(inp['L_fw'][0], None, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, None, VB_dec_init_state)
        elif signal == 'describe':
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            if appriori_len:
                L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
            else:
                L_output = self.lang_decoder(inp['L_fw'][0], None, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, None, VB_dec_init_state)
        elif signal == 'repeat language':
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']),1,1), inp["B_fw"][0, :, :]]
            if appriori_len:
                L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
            else:
                L_output = self.lang_decoder(inp['L_fw'][0], None, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, None, VB_dec_init_state)
        else:
            VB_input_f = inp['VB_fw']
            if appriori_len:
                L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, len(inp['B_fw']), VB_dec_init_state)
            else:
                L_output = self.lang_decoder(inp['L_fw'][0], None, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, None, VB_dec_init_state)
        return L_output, B_output

    def inference(self, inp, signal, appriori_len=True):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_fw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        if self.lang_enc_type == 'LSTM':
            encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        elif self.lang_enc_type == 'BERT':
            encoded_lang = self.lang_encoder(lang_inp)
        elif self.lang_enc_type == 'WordEmbedding':
            encoded_lang = self.word_embedder(lang_inp.argmax(axis=-1)).permute(1, 0, 2)
        else:
            encoded_lang = lang_inp.permute(1, 0, 2).float()

        if self.act_enc_type == 'LSTM':
            encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        else:
            encoded_act = VB_input.permute(1, 0, 2).float()

        h=self.hidden(encoded_lang, encoded_act, None, None)
        h = h.mean(1)

        L_dec_init_state = self.initial_lang(h)
        VB_dec_init_state = self.initial_act(h)
        #B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)

        if signal=='execute' or signal == 'repeat action':
            if appriori_len:
                L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)
            else:
                L_output = self.lang_decoder(inp['L_fw'][0], None, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, None, VB_dec_init_state)
        else:
            #self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)
            if appriori_len:
                L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)
            else:
                L_output = self.lang_decoder(inp['L_fw'][0], None, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, None, VB_dec_init_state)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot, B_output

    def extract_representations(self, inp, signal):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = [inp["V_fw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_fw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0
            l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        if self.lang_enc_type == 'LSTM':
            encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        elif self.lang_enc_type == 'BERT':
            encoded_lang = self.lang_encoder(lang_inp)
        elif self.lang_enc_type == 'WordEmbedding':
            encoded_lang = self.word_embedder(lang_inp.argmax(axis=-1)).permute(1, 0, 2)
        else:
            encoded_lang = lang_inp.permute(1, 0, 2).float()

        if self.act_enc_type == 'LSTM':
            encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        else:
            encoded_act = VB_input.permute(1, 0, 2).float()

        h=self.hidden(encoded_lang, encoded_act, None, None)

        return h#return encoded_lang, h

    def inference_conflict(self, inp, signal, appriori_len=True):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim + 4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            VB_input_f = [inp['VB_fw'][0][0].repeat(len(inp['V_fw']), 1, 1), inp['VB_fw'][1]]
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim + 2] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:, self.params.L_input_dim] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            VB_input_f = [inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0, :, :]]
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:, self.params.L_input_dim + 1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            #VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
            #           inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            VB_input_f = inp['VB_fw']
            lang_in_length = inp['L_len'].int().numpy()

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        if self.lang_enc_type == 'LSTM':
            encoded_lang = self.lang_encoder(lang_inp, lang_in_length + 1, True)
        elif self.lang_enc_type == 'BERT':
            encoded_lang = self.lang_encoder(lang_inp)
        elif self.lang_enc_type == 'WordEmbedding':
            encoded_lang = self.word_embedder(lang_inp.argmax(axis=-1)).permute(1, 0, 2)
        else:
            encoded_lang = lang_inp.permute(1, 0, 2).float()

        if self.act_enc_type == 'LSTM':
            encoded_act = self.action_encoder(VB_input, inp['V_len'].int().numpy())
        else:
            encoded_act = VB_input.permute(1, 0, 2).float()

        h=self.hidden(encoded_lang, encoded_act, None, None)
        h = h.mean(1)

        L_dec_init_state = self.initial_lang(h)
        VB_dec_init_state = self.initial_act(h)
        #B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)

        if signal=='execute' or signal == 'repeat action':
            if appriori_len:
                L_output = self.lang_decoder(inp['L_fw'][0], 2, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)
            else:
                L_output = self.lang_decoder(inp['L_fw'][0], None, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, None, VB_dec_init_state)
        else:
            #self.lang_decoder(inp['L_fw'][0], int(inp['L_len'].item()), L_dec_init_state, True)
            if appriori_len:
                L_output = self.lang_decoder(inp['L_fw'][0], len(inp['L_fw']), L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, int(inp['B_len'].item()), VB_dec_init_state)
            else:
                L_output = self.lang_decoder(inp['L_fw'][0], None, L_dec_init_state, True)
                B_output = self.action_decoder(VB_input_f, None, VB_dec_init_state)
        max_ind = torch.argmax(L_output, -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot, B_output

class CrossmodalTransformerDecoderTransformer(nn.Module):
    def __init__(self, params, word_embedding=False, app_length=True):
        super(CrossmodalTransformerDecoderTransformer, self).__init__()
        from crossmodal_transformer import Visual_Ling_Attn as CMTransformer
        from crossmodal_transformer import LanguageDecoderLayer as LangDecoder
        from crossmodal_transformer import ActionDecoderLayer as ActDecoder
        self.params = params

        if word_embedding:
            self.word_embedding = True
            self.word_embedder = Embedder(self.params.L_input_dim + 5, emb_dim=params.L_num_units)
        else:
            self.word_embedding = False

        self.hidden = CMTransformer(self.params)

        #self.initial_lang = nn.Linear(self.params.hidden_dim, self.params.L_num_units*self.params.L_num_layers*2)
        #self.initial_act = nn.Linear(self.params.hidden_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        self.lang_decoder = LangDecoder(self.params, self.hidden.ins_fc)
        self.action_decoder = ActDecoder(self.params, self.hidden.vis_fc, appriori_length=app_length)

    def forward(self, inp, signal, appriori_len=True):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim+4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim+2] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim+1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input=torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) *\
                     inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        if self.word_embedding:
            encoded_lang = self.word_embedder(lang_inp.argmax(axis=-1)).permute(1,0,2)
        else:
            encoded_lang = lang_inp.permute(1,0,2).float()

        encoded_act = VB_input.permute(1,0,2).float()#encoded_act_orig = VB_input.permute(1,0,2).float()#
            #encoded_act = []
            #ts_interval = int(encoded_act_orig.shape[1] / encoded_lang.shape[1])
            #for ts in range(0, encoded_lang.shape[1]):
            #    encoded_act.append(encoded_act_orig[:, ts*ts_interval, :])
            #encoded_act = torch.stack(encoded_act, dim=1)

        h = self.hidden(encoded_lang, encoded_act, None, None)
        #h = h.mean(1)

        #L_dec_init_state = self.initial_lang(h)
        #VB_dec_init_state = self.initial_act(h)
        mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, VB_input.shape[0]-1, VB_input.shape[0]-1).bool(), diagonal=1).cuda()
        mask_enc_att = None
        if signal == 'repeat both':
            #VB_input_f = inp['VB_fw']
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, l_fw_ndim[:-1].shape[0],
                                                       l_fw_ndim[:-1].shape[0]).bool(), diagonal=1).cuda()
            if appriori_len:
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1,0,2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1,0,2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1,0,2), None,  h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1,0,2), None, h, mask_self_att_act, mask_enc_att)
        elif signal == 'describe':
            VB_input_f = torch.cat((inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0].repeat(len(inp['B_fw']), 1, 1)),axis=-1)
            L_tar = torch.cat((inp['L_fw'][:-1], torch.zeros(inp['L_fw'].size()[0]-1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_tar.shape[0],
                                                       L_tar.shape[0]).bool(), diagonal=1).cuda()
            if appriori_len:
                L_output = self.lang_decoder(L_tar.permute(1,0,2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1,0,2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(L_tar.permute(1,0,2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1,0,2), None, h, mask_self_att_act, mask_enc_att)
        elif signal == 'repeat language':
            VB_input_f = torch.cat((inp["V_fw"][0].repeat(len(inp['V_fw']),1,1), inp["B_fw"][0].repeat(len(inp['B_fw']),1,1)),axis=-1)
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, l_fw_ndim[:-1].shape[0],
                                                       l_fw_ndim[:-1].shape[0]).bool(), diagonal=1).cuda()
            if appriori_len:
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1,0,2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1,0,2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1,0,2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1,0,2), None, h, mask_self_att_act, mask_enc_att)
        elif signal == 'repeat action':
            L_tar = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_tar.shape[0],
                                                       L_tar.shape[0]).bool(), diagonal=1).cuda()
            #VB_input_f = inp['VB_fw']
            if appriori_len:
                L_output = self.lang_decoder(L_tar.permute(1,0,2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1,0,2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(L_tar.permute(1,0,2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1,0,2), None, h, mask_self_att_act, mask_enc_att)
        else:
            # VB_input_f = inp['VB_fw']

            L_tar = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_tar.shape[0],
                                                       L_tar.shape[0]).bool(), diagonal=1).cuda()
            if appriori_len:
                L_output = self.lang_decoder(L_tar.permute(1, 0, 2), h, mask_self_att_lang,
                                             mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(L_tar.permute(1, 0, 2), None, h, mask_self_att_lang,
                                             mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1, 0, 2), None, h, mask_self_att_act, mask_enc_att)
        return L_output, B_output

    def inference(self, inp, signal, appriori_len=True):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0,:,self.params.L_input_dim+4] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input = torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0,:,self.params.L_input_dim+2] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2]+5), requires_grad=True).to('cuda')
            VB_input = torch.cat([inp['V_fw'], inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2]+5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')
            lang_in_length = inp['L_len'].int().numpy() - (inp['L_len'].int().numpy() - 1)
        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim+1] = 1.0
            #VB_input = torch.zeros((inp['V_fw'].size()[0], inp['V_fw'].size()[1], inp['V_fw'].size()[2]+inp['B_fw'].size()[2])).to('cuda')
            VB_input=torch.cat((inp['V_fw'][0].repeat(len(inp['V_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) *\
                     inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, inp['V_fw'].size()[-1] + inp['B_fw'].size()[-1])
            lang_in_length = inp['L_len'].int().numpy()
        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        if self.word_embedding:
            encoded_lang = self.word_embedder(lang_inp.argmax(axis=-1)).permute(1,0,2)
        else:
            encoded_lang = lang_inp.permute(1,0,2).float()


        encoded_act = VB_input.permute(1,0,2).float()

        h = self.hidden(encoded_lang, encoded_act, None, None)
        #h = h.mean(1)

        #L_dec_init_state = self.initial_lang(h)
        #VB_dec_init_state = self.initial_act(h)
        #mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, 1, 1).bool(), diagonal=1).cuda()
        mask_enc_att = None
        if signal == 'describe':
            #VB_input_f = torch.cat((inp["V_bw"][0].repeat(len(inp['V_fw']), 1, 1), inp["B_bw"][0].repeat(len(inp['B_fw']), 1, 1)),axis=-1)
            B_out = torch.cat((inp["V_bw"][0].unsqueeze(0), inp["B_bw"][0].unsqueeze(0)), axis=-1)
            L_out = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            if appriori_len:
                for i in range(int(inp['L_len'].item())-1):
                    mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                               L_out.shape[0]).bool(), diagonal=1).cuda()
                    L_output = self.lang_decoder(L_out.permute(1,0,2), h, mask_self_att_lang, mask_enc_att)
                    #use one hot encoded version of the output
                    #max_ind = torch.argmax(L_output[-1, :, :self.params.L_input_dim], -1)
                    #L_output = nn.functional.one_hot(max_ind, self.params.L_input_dim).unsqueeze(0)
                    #L_output = torch.cat((L_output, torch.zeros(1, L_output.size()[1], 5).to('cuda')),axis=-1).to('cuda')
                    L_output = torch.cat((L_output[-1].unsqueeze(0), torch.zeros(1, L_output.size()[1], 5).to('cuda')), axis=-1).to('cuda')
                    L_out = torch.cat((L_out,L_output), axis=0)
                for i in range(int(inp['B_len'].item()) - 1):
                    mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                                                   diagonal=1).cuda()
                    B_output = self.action_decoder(B_out.permute(1,0,2), h, mask_self_att_act, mask_enc_att)
                    B_output = torch.cat((inp["V_bw"][0].unsqueeze(0), B_output[-1].unsqueeze(0)), axis=-1)
                    B_out = torch.cat((B_out,B_output), axis=0)
            else:   ## code this later
                mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                    diagonal=1).cuda()
                mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                           L_out.shape[0]).bool(), diagonal=1).cuda()
                L_output = self.lang_decoder(L_out.permute(1,0,2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(B_out[:-1].permute(1,0,2), None, h, mask_self_att_act, mask_enc_att)
        elif signal == 'repeat language':
            #VB_input_f = torch.cat((inp["V_fw"][0].repeat(len(inp['V_fw']),1,1), inp["B_fw"][0].repeat(len(inp['B_fw']),1,1)),axis=-1)
            B_out = torch.cat((inp["V_fw"][0].unsqueeze(0), inp["B_fw"][0].unsqueeze(0)), axis=-1)
            L_out = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            if appriori_len:
                for i in range(int(inp['L_len'].item())-1):
                    mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                               L_out.shape[0]).bool(), diagonal=1).cuda()
                    L_output = self.lang_decoder(L_out.permute(1,0,2), h, mask_self_att_lang, mask_enc_att)
                    L_output = torch.cat((L_output[-1].unsqueeze(0), torch.zeros(1, L_output.size()[1], 5).to('cuda')), axis=-1).to('cuda')
                    L_out = torch.cat((L_out,L_output), axis=0)
                for i in range(int(inp['B_len'].item()) - 1):
                    mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                                                   diagonal=1).cuda()
                    B_output = self.action_decoder(B_out.permute(1,0,2), h, mask_self_att_act, mask_enc_att)
                    B_output = torch.cat((inp["V_fw"][0].unsqueeze(0), B_output[-1].unsqueeze(0)), axis=-1)
                    B_out = torch.cat((B_out,B_output), axis=0)
            else: ## code later
                mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                           L_out.shape[0]).bool(), diagonal=1).cuda()
                mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                    diagonal=1).cuda()
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1,0,2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(B_out[:-1].permute(1,0,2), None, h, mask_self_att_act, mask_enc_att)
        elif signal == 'repeat action':
            L_out = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            #VB_input_f = inp['VB_fw']
            B_out = torch.cat((inp["V_fw"][0].unsqueeze(0), inp["B_fw"][0].unsqueeze(0)), axis=-1)
            if appriori_len:
                for i in range(1):
                    mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                               L_out.shape[0]).bool(), diagonal=1).cuda()
                    L_output = self.lang_decoder(L_out.permute(1,0,2), h, mask_self_att_lang, mask_enc_att)
                    L_output = torch.cat((L_output[-1].unsqueeze(0), torch.zeros(1, L_output.size()[1], 5).to('cuda')), axis=-1).to('cuda')
                    L_out = torch.cat((L_out,L_output), axis=0)
                for i in range(int(inp['B_len'].item()) - 1):
                    mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                                                   diagonal=1).cuda()
                    B_output = self.action_decoder(B_out.permute(1,0,2), h, mask_self_att_act, mask_enc_att)
                    B_output = torch.cat((inp["V_fw"][i+1].unsqueeze(0), B_output[-1].unsqueeze(0)), axis=-1)
                    B_out = torch.cat((B_out,B_output), axis=0)
            else: ## code later
                mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                           L_out.shape[0]).bool(), diagonal=1).cuda()
                mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                    diagonal=1).cuda()
                L_output = self.lang_decoder(L_out.permute(1,0,2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(B_out[:-1].permute(1,0,2), None, h, mask_self_att_act, mask_enc_att)
        else:
            # VB_input_f = inp['VB_fw']
            B_out = torch.cat((inp["V_fw"][0].unsqueeze(0), inp["B_fw"][0].unsqueeze(0)), axis=-1)
            L_out = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            if appriori_len:
                for i in range(1):
                    mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                               L_out.shape[0]).bool(), diagonal=1).cuda()
                    L_output = self.lang_decoder(L_out.permute(1,0,2), h, mask_self_att_lang, mask_enc_att)
                    L_output = torch.cat((L_output[-1].unsqueeze(0), torch.zeros(1, L_output.size()[1], 5).to('cuda')), axis=-1).to('cuda')
                    L_out = torch.cat((L_out,L_output), axis=0)
                for i in range(int(inp['B_len'].item()) - 1):
                    mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                                                   diagonal=1).cuda()
                    B_output = self.action_decoder(B_out.permute(1,0,2), h, mask_self_att_act, mask_enc_att)
                    B_output = torch.cat((inp["V_fw"][i+1].unsqueeze(0), B_output[-1].unsqueeze(0)), axis=-1)
                    B_out = torch.cat((B_out,B_output), axis=0)
            else: ## code later
                mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                           L_out.shape[0]).bool(), diagonal=1).cuda()
                mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                    diagonal=1).cuda()
                L_output = self.lang_decoder(L_out.permute(1, 0, 2), None, h, mask_self_att_lang,
                                             mask_enc_att)
                B_output = self.action_decoder(B_out[:-1].permute(1, 0, 2), None, h, mask_self_att_act, mask_enc_att)
        max_ind = torch.argmax(L_out[1:,:,:self.params.L_input_dim], -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot.squeeze(1), B_out[1:,:,self.params.VB_input_dim-self.params.B_input_dim:].squeeze(1)

class CrossmodalTransformerDecoderTransformerResNet(nn.Module):
    def __init__(self, params, word_embedding=False, app_length=True):
        super(CrossmodalTransformerDecoderTransformerResNet, self).__init__()
        from crossmodal_transformer import Visual_Ling_Attn as CMTransformer
        from crossmodal_transformer import LanguageDecoderLayer as LangDecoder
        from crossmodal_transformer import ActionDecoderLayer as ActDecoder
        self.params = params

        self.vis_feat_extractor = ResNetFeatExtractor(18, self.params.B_max_length)#EfficientNetFeatExtractor('b0', self.params.B_max_length)

        if word_embedding:
            self.word_embedding = True
            self.word_embedder = Embedder(self.params.L_input_dim + 5, emb_dim=params.L_num_units)
        else:
            self.word_embedding = False

        self.hidden = CMTransformer(self.params)

        # self.initial_lang = nn.Linear(self.params.hidden_dim, self.params.L_num_units*self.params.L_num_layers*2)
        # self.initial_act = nn.Linear(self.params.hidden_dim, self.params.VB_num_units*self.params.VB_num_layers*2)

        self.lang_decoder = LangDecoder(self.params, self.hidden.ins_fc)
        self.action_decoder = ActDecoder(self.params, self.hidden.vis_fc, appriori_length=app_length)

    def forward(self, inp, signal, appriori_len=True):
        if signal == 'repeat both':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            visual_features = self.vis_feat_extractor(inp['images']).transpose(0, 1)
            VB_input = visual_features#torch.cat([visual_features, inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 3] = 1.0

        elif signal == 'repeat language':
            l_fw_ndim = torch.cat(
                (inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 4] = 1.0
            for idx in range(len(inp['images'])):
                 inp['images'][idx] = inp['images'][idx][0].unsqueeze(0)
            visual_features = self.vis_feat_extractor(inp['images'], True).transpose(0, 1)
            VB_input = torch.cat((visual_features.repeat(len(inp['B_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, visual_features.size()[-1] + inp['B_fw'].size()[-1])

            # visual_features.repeat(len(inp['B_fw']), 1, 1)#
        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to(
                'cuda')
            visual_features = self.vis_feat_extractor(inp['images']).transpose(0, 1)
            VB_input = torch.cat([visual_features, inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 2] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat(
                    (inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                    'cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2] + 5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')

        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to(
                'cuda')
            visual_features = self.vis_feat_extractor(inp['images']).transpose(0, 1)
            VB_input = torch.cat([visual_features, inp['B_fw']], dim=2)#visual_features#
            signalrow[0, :, self.params.L_input_dim] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat(
                    (inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                    'cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2] + 5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')

        else:
            l_fw_ndim = torch.cat(
                (inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to(
                'cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 1] = 1.0
            #for idx in range(len(inp['images'])):
                #inp['images'][idx] = inp['images'][idx][0].unsqueeze(0)
            visual_features = self.vis_feat_extractor(inp['images']).transpose(0, 1)
            VB_input = torch.cat((visual_features[0].repeat(len(inp['B_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, visual_features[0].size()[-1] + inp['B_fw'].size()[-1])

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        if self.word_embedding:
            encoded_lang = self.word_embedder(lang_inp.argmax(axis=-1)).permute(1, 0, 2)
        else:
            encoded_lang = lang_inp.permute(1, 0, 2).float()

        encoded_act = VB_input.permute(1, 0, 2).float()  # encoded_act_orig = VB_input.permute(1,0,2).float()#
        # encoded_act = []
        # ts_interval = int(encoded_act_orig.shape[1] / encoded_lang.shape[1])
        # for ts in range(0, encoded_lang.shape[1]):
        #    encoded_act.append(encoded_act_orig[:, ts*ts_interval, :])
        # encoded_act = torch.stack(encoded_act, dim=1)

        h = self.hidden(encoded_lang, encoded_act, None, None)
        # h = h.mean(1)

        # L_dec_init_state = self.initial_lang(h)
        # VB_dec_init_state = self.initial_act(h)
        mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads,
                                                  VB_input.shape[0] - 1, VB_input.shape[0] - 1).bool(),diagonal=1).cuda()
        mask_enc_att = None
        if signal == 'repeat both':
            # VB_input_f = inp['VB_fw']
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, l_fw_ndim[:-1].shape[0],
                                                       l_fw_ndim[:-1].shape[0]).bool(), diagonal=1).cuda()
            if appriori_len:
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
        elif signal == 'describe':
            VB_input_f = torch.cat((visual_features[-1].repeat(len(inp['B_fw']), 1, 1), inp["B_bw"][0].repeat(len(inp['B_fw']), 1, 1)), axis=-1)#visual_features[-1].repeat(len(inp['B_fw']), 1, 1)
            L_tar = torch.cat(
                (inp['L_fw'][:-1], torch.zeros(inp['L_fw'].size()[0] - 1, inp['L_fw'].size()[1], 5).to('cuda')),
                axis=-1).to(
                'cuda')
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_tar.shape[0],
                                                       L_tar.shape[0]).bool(), diagonal=1).cuda()
            if appriori_len:
                L_output = self.lang_decoder(L_tar.permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(L_tar.permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1, 0, 2), h, mask_self_att_act,
                                               mask_enc_att)
        elif signal == 'repeat language':
            VB_input_f = torch.cat((visual_features.repeat(len(inp['B_fw']), 1, 1), inp["B_fw"][0].repeat(len(inp['B_fw']), 1, 1)), axis=-1)#visual_features.repeat(len(inp['B_fw']), 1, 1)#
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, l_fw_ndim[:-1].shape[0],
                                                       l_fw_ndim[:-1].shape[0]).bool(), diagonal=1).cuda()
            if appriori_len:
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1, 0, 2), h, mask_self_att_act,
                                               mask_enc_att)
        elif signal == 'repeat action':
            L_tar = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to(
                'cuda')
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_tar.shape[0],
                                                       L_tar.shape[0]).bool(), diagonal=1).cuda()
            # VB_input_f = inp['VB_fw']
            if appriori_len:
                L_output = self.lang_decoder(L_tar.permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(L_tar.permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(VB_input[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
        else:
            # VB_input_f = inp['VB_fw']
            VB_input_f = torch.cat([visual_features, inp['B_fw']], dim=2)
            L_tar = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to(
                'cuda')
            mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_tar.shape[0],
                                                       L_tar.shape[0]).bool(), diagonal=1).cuda()
            if appriori_len:
                L_output = self.lang_decoder(L_tar.permute(1, 0, 2), h, mask_self_att_lang,
                                             mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
            else:
                L_output = self.lang_decoder(L_tar.permute(1, 0, 2), h, mask_self_att_lang,
                                             mask_enc_att)
                B_output = self.action_decoder(VB_input_f[:-1].permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
        return L_output, B_output

    def inference(self, inp, signal, appriori_len=True):
        if signal == 'repeat language':
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 4] = 1.0
            for idx in range(len(inp['images'])):
                 inp['images'][idx] = inp['images'][idx][0].unsqueeze(0)
            visual_features = self.vis_feat_extractor(inp['images'], True).transpose(0, 1)
            VB_input = torch.cat((visual_features.repeat(len(inp['B_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, visual_features.size()[-1] + inp['B_fw'].size()[-1])

        elif signal == 'repeat action':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
            visual_features = self.vis_feat_extractor(inp['images']).transpose(0, 1)
            VB_input = torch.cat([visual_features, inp['B_fw']], dim=2)
            signalrow[0, :, self.params.L_input_dim + 2] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2] + 5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')

        elif signal == 'describe':
            signalrow = torch.zeros((1, inp['L_fw'].size()[1], inp['L_fw'].size()[2] + 5), requires_grad=True).to('cuda')
            visual_features = self.vis_feat_extractor(inp['images']).transpose(0, 1)
            VB_input = torch.cat([visual_features, inp['B_fw']], dim=2)#visual_features#
            signalrow[0, :, self.params.L_input_dim] = 1.0
            if appriori_len:
                l_fw_ndim = torch.cat((inp['L_fw'][-1].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            else:
                l_fw_eos = torch.zeros((1, inp['L_fw'].shape[1], inp['L_fw'].shape[2] + 5))
                l_fw_eos[:, :, 1] = 1
                l_fw_ndim = l_fw_eos.to('cuda')

        else:
            l_fw_ndim = torch.cat((inp['L_fw'], torch.zeros(inp['L_fw'].size()[0], inp['L_fw'].size()[1], 5).to('cuda')), axis=-1).to('cuda')
            signalrow = torch.zeros((1, l_fw_ndim.size()[1], l_fw_ndim.size()[2]), requires_grad=True).to('cuda')
            signalrow[0, :, self.params.L_input_dim + 1] = 1.0
            #inp['images_inp'] = []
            #for idx in range(len(inp['images'])):
            #    inp['images_inp'].append(inp['images'][idx][0].unsqueeze(0))
            visual_features = self.vis_feat_extractor(inp['images']).transpose(0, 1)
            VB_input = torch.cat((visual_features[0].repeat(len(inp['B_fw']), 1, 1), inp['B_fw'][0].repeat(len(inp['B_fw']), 1, 1)), dim=2) * \
                       inp['B_bin'][:, :, 0].unsqueeze(-1).repeat(1, 1, visual_features[0].size()[-1] + inp['B_fw'].size()[-1])

        lang_inp = torch.cat((signalrow, l_fw_ndim), axis=0).to('cuda')

        if self.word_embedding:
            encoded_lang = self.word_embedder(lang_inp.argmax(axis=-1)).permute(1, 0, 2)
        else:
            encoded_lang = lang_inp.permute(1, 0, 2).float()

        encoded_act = VB_input.permute(1, 0, 2).float()
        h = self.hidden(encoded_lang, encoded_act, None, None)

        mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, VB_input.shape[0] - 1, VB_input.shape[0] - 1).bool(), diagonal=1).cuda()
        mask_enc_att = None

        if signal == 'describe':
            #VB_input_f = torch.cat((visual_features[-1].repeat(len(inp['B_fw']), 1, 1), inp["B_bw"][0].repeat(len(inp['B_fw']), 1, 1)), axis=-1)
            B_out = torch.cat((visual_features[-1].unsqueeze(0), inp["B_bw"][0].unsqueeze(0)), axis=-1)#visual_features[-1].unsqueeze(0)#
            L_out = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),axis=-1).to('cuda')
            if appriori_len:
                for i in range(int(inp['L_len'].item()) - 1):
                    mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                               L_out.shape[0]).bool(), diagonal=1).cuda()
                    L_output = self.lang_decoder(L_out.permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                    # use one hot encoded version of the output
                    # max_ind = torch.argmax(L_output[-1, :, :self.params.L_input_dim], -1)
                    # L_output = nn.functional.one_hot(max_ind, self.params.L_input_dim).unsqueeze(0)
                    # L_output = torch.cat((L_output, torch.zeros(1, L_output.size()[1], 5).to('cuda')),axis=-1).to('cuda')
                    L_output = torch.cat((L_output[-1].unsqueeze(0), torch.zeros(1, L_output.size()[1], 5).to('cuda')),
                                         axis=-1).to('cuda')
                    L_out = torch.cat((L_out, L_output), axis=0)
                for i in range(int(inp['B_len'].item()) - 1):
                    mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),diagonal=1).cuda()
                    B_output = self.action_decoder(B_out.permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
                    B_output = torch.cat((visual_features[-1].unsqueeze(0), B_output[-1].unsqueeze(0)), axis=-1)#visual_features[-1].unsqueeze(0)#
                    B_out = torch.cat((B_out, B_output), axis=0)
            else: # code later
                mask_self_att_act = torch.triu(
                    torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                    diagonal=1).cuda()
                mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                           L_out.shape[0]).bool(), diagonal=1).cuda()
                L_output = self.lang_decoder(L_out.permute(1, 0, 2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(B_out[:-1].permute(1, 0, 2), None, h, mask_self_att_act, mask_enc_att)

        elif signal == 'repeat language':
            #VB_input_f = torch.cat((visual_features.repeat(len(inp['B_fw']), 1, 1), inp["B_fw"][0].repeat(len(inp['B_fw']), 1, 1)), axis=-1)
            B_out = torch.cat((visual_features[0].unsqueeze(0), inp["B_fw"][0].unsqueeze(0)), axis=-1)
            L_out = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),axis=-1).to('cuda')

            if appriori_len:
                for i in range(int(inp['L_len'].item()) - 1):
                    mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                               L_out.shape[0]).bool(), diagonal=1).cuda()
                    L_output = self.lang_decoder(L_out.permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                    L_output = torch.cat((L_output[-1].unsqueeze(0), torch.zeros(1, L_output.size()[1], 5).to('cuda')),
                                         axis=-1).to('cuda')
                    L_out = torch.cat((L_out, L_output), axis=0)
                for i in range(int(inp['B_len'].item()) - 1):
                    mask_self_att_act = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                        diagonal=1).cuda()
                    B_output = self.action_decoder(B_out.permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
                    B_output = torch.cat((visual_features[0].unsqueeze(0), B_output[-1].unsqueeze(0)), axis=-1)
                    B_out = torch.cat((B_out, B_output), axis=0)
            else: ## code later
                mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                           L_out.shape[0]).bool(), diagonal=1).cuda()
                mask_self_att_act = torch.triu(
                    torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                    diagonal=1).cuda()
                L_output = self.lang_decoder(l_fw_ndim[:-1].permute(1, 0, 2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(B_out[:-1].permute(1, 0, 2), None, h, mask_self_att_act, mask_enc_att)

        elif signal == 'repeat action':
            L_out = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),
                              axis=-1).to(
                'cuda')
            # VB_input_f = inp['VB_fw']
            B_out = torch.cat((visual_features[0].unsqueeze(0), inp["B_fw"][0].unsqueeze(0)), axis=-1)
            if appriori_len:
                for i in range(1):
                    mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                               L_out.shape[0]).bool(), diagonal=1).cuda()
                    L_output = self.lang_decoder(L_out.permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                    L_output = torch.cat((L_output[-1].unsqueeze(0), torch.zeros(1, L_output.size()[1], 5).to('cuda')),axis=-1).to('cuda')
                    L_out = torch.cat((L_out, L_output), axis=0)
                for i in range(int(inp['B_len'].item()) - 1):
                    mask_self_att_act = torch.triu(
                        torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                        diagonal=1).cuda()
                    B_output = self.action_decoder(B_out.permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
                    B_output = torch.cat((visual_features[i + 1].unsqueeze(0), B_output[-1].unsqueeze(0)), axis=-1)
                    B_out = torch.cat((B_out, B_output), axis=0)
            else:  ## code later
                mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                           L_out.shape[0]).bool(), diagonal=1).cuda()
                mask_self_att_act = torch.triu(
                    torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                    diagonal=1).cuda()
                L_output = self.lang_decoder(L_out.permute(1, 0, 2), None, h, mask_self_att_lang, mask_enc_att)
                B_output = self.action_decoder(B_out[:-1].permute(1, 0, 2), None, h, mask_self_att_act, mask_enc_att)

        else:
            # VB_input_f = inp['VB_fw']
            B_out = torch.cat((visual_features[0].unsqueeze(0), inp["B_fw"][0].unsqueeze(0)), axis=-1)
            L_out = torch.cat((inp['L_fw'][0].unsqueeze(0), torch.zeros(1, inp['L_fw'].size()[1], 5).to('cuda')),axis=-1).to('cuda')
            if appriori_len:
                for i in range(1):
                    mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                               L_out.shape[0]).bool(), diagonal=1).cuda()
                    L_output = self.lang_decoder(L_out.permute(1, 0, 2), h, mask_self_att_lang, mask_enc_att)
                    L_output = torch.cat((L_output[-1].unsqueeze(0), torch.zeros(1, L_output.size()[1], 5).to('cuda')),
                                         axis=-1).to('cuda')
                    L_out = torch.cat((L_out, L_output), axis=0)
                for i in range(int(inp['B_len'].item()) - 1):
                    mask_self_att_act = torch.triu(
                        torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                        diagonal=1).cuda()
                    B_output = self.action_decoder(B_out.permute(1, 0, 2), h, mask_self_att_act, mask_enc_att)
                    B_output = torch.cat((visual_features[i + 1].unsqueeze(0), B_output[-1].unsqueeze(0)), axis=-1)
                    B_out = torch.cat((B_out, B_output), axis=0)
            else:  ## code later
                mask_self_att_lang = torch.triu(torch.ones(h.shape[0], self.params.T_num_heads, L_out.shape[0],
                                                           L_out.shape[0]).bool(), diagonal=1).cuda()
                mask_self_att_act = torch.triu(
                    torch.ones(h.shape[0], self.params.T_num_heads, B_out.shape[0], B_out.shape[0]).bool(),
                    diagonal=1).cuda()
                L_output = self.lang_decoder(L_out.permute(1, 0, 2), None, h, mask_self_att_lang,
                                             mask_enc_att)
                B_output = self.action_decoder(B_out[:-1].permute(1, 0, 2), None, h, mask_self_att_act, mask_enc_att)

        max_ind = torch.argmax(L_out[1:, :, :self.params.L_input_dim], -1)
        one_hot = nn.functional.one_hot(max_ind, inp['L_fw'].size()[-1])

        return one_hot.squeeze(1), B_out[1:, :, self.params.VB_input_dim - self.params.B_input_dim:].squeeze(1)
