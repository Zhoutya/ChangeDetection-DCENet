# -*- coding: utf-8 -*-
import time
import datetime
import math
import os
from torch.utils.data import DataLoader
import contextlib
from torch.cuda.amp import autocast

from model.losses import *


def adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, step_index):
    # Adjust the learning rate in stages
    if epoch < 1:
        lr = 0.0001 * lr_init
    elif epoch <= step_index[0]:
        lr = lr_init
    elif epoch <= step_index[1]:
        lr = lr_init * lr_gamma
    elif epoch <= step_index[2]:
        lr = lr_init * lr_gamma ** 2
    elif epoch > step_index[2]:
        lr = lr_init * lr_gamma ** 3

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def train_ULiter(trl_data, tru_data, model, optimizer, device, cfg):
    torch.autograd.set_detect_anomaly(True)
    num_workers = cfg['workers_num']
    save_folder = cfg['save_folder']
    save_name = cfg['save_name']

    lr_init = cfg['lr']
    lr_gamma = cfg['lr_gamma']
    lr_step = cfg['lr_step']
    lr_adjust = cfg['lr_adjust']

    epoch_size = cfg['epoch']
    batch_size_l = cfg['batch_size_l']
    batch_size_u = cfg['batch_size_u']

    '''# Load the model and start training'''
    model.train()
    start_epoch = 0
    centers_batch0 = torch.randn(1, 2)
    centers_batch1 = torch.randn(1, 2)
    batch_num = math.ceil(len(trl_data) / batch_size_l) - 1  # math.ceil 向上取整
    print('Sample: label{}/{} || unlabel: {}/{} ;batch_num: {}'.format(len(trl_data), batch_size_l, len(tru_data),
                                                                       batch_size_u, batch_num))

    amp_cm = autocast if 1 else contextlib.nullcontext
    print('start training...')
    for epoch in range(start_epoch + 1, epoch_size):
        epoch_time0 = time.time()
        epoch_loss, epoch_sup_loss, epoch_unsup_loss, epoch_cl_loss = 0, 0, 0, 0

        batch_datal = DataLoader(trl_data, batch_size_l, shuffle=True, num_workers=num_workers, pin_memory=True,
                                 drop_last=True)
        batch_datau = DataLoader(tru_data, batch_size_u, shuffle=True, num_workers=num_workers, pin_memory=True,
                                 drop_last=True)

        if lr_adjust:
            lr = adjust_lr_sub(lr_init, lr_gamma, optimizer, epoch, lr_step)
        else:
            lr = lr_init

        # Enter M labelled data and N unlabelled data
        for batch_idx, data in enumerate(zip(batch_datal, batch_datau)):
            batch_time0 = time.time()
            xl1, xl2, gtl, indicesl = data[0]
            xu1, xu2, gtu, indicesu = data[1]
            xl1, xl2, xu1, xu2 = xl1.to(device), xl2.to(device), xu1.to(device), xu2.to(device)
            gtl = gtl.to(device)

            with amp_cm():
                # sup+CL+KL
                logits_x_l, logits_x_u, L_cl, L_unsup = model(xl1, xl2, xu1, xu2)
                L_l_sup = ce_loss(logits_x_l, gtl.long(), reduction='mean')
                total_loss = L_l_sup + L_unsup + L_cl

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # estimated time of Arrival
            batch_time = time.time() - batch_time0
            batch_eta = batch_time * (batch_num - batch_idx)
            epoch_eta = int(batch_time * (epoch_size - epoch) * batch_num + batch_eta)

            epoch_loss += total_loss.item()
            epoch_sup_loss += L_l_sup.item()
            epoch_cl_loss += L_cl.item()
            epoch_unsup_loss += L_unsup.item()

        epoch_time = time.time() - epoch_time0
        epoch_eta = int(epoch_time * (epoch_size - epoch))

        print('Epoch: {}/{} || lr: {} || total_loss: {:.4f} sup_loss: {:.4f} unsup_loss ：{:.4f} cl_loss ：{:.4f} || '
              'Epoch time: {:.4f}s || Epoch ETA: {}'
              .format(epoch, epoch_size, lr, epoch_loss / batch_num, epoch_sup_loss / batch_num,
                      epoch_unsup_loss / batch_num, epoch_cl_loss / batch_num,
                      epoch_time, str(datetime.timedelta(seconds=epoch_eta))))

    # Save the final model
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)  # 递归创建目录
    save_model = dict(
        model=model.state_dict(),
        epoch=epoch_size
    )
    torch.save(save_model, os.path.join(save_folder, save_name + '_Final.pth'))
