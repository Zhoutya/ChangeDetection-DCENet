# -*- coding: utf-8 -*-
import os
import torch.optim as optim
import imageio
import scipy.io as io

import configs.configs as cfg
from data.HSICD_data import HSICD_data
from data.get_train_test_set import get_train_test_set as get_set
from model.mainmodel import *
from tools.utils import *
from tools.train_utils import *
from tools.test_utils import *


def main():
    current_dataset = cfg.current_dataset
    current_model = cfg.current_model
    model_name = current_dataset + current_model
    cfg_data = cfg.data
    cfg_trainUL = cfg.train['train_UL_model']
    cfg_optim = cfg.train['optimizer']
    cfg_test = cfg.test
    in_fea_num, feature_dim, class_num = cfg.model[current_dataset]

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Data import and data set partition
    data_sets = get_set(cfg_data)
    img_gt = data_sets['img_gt']
    trl_data = HSICD_data(data_sets, cfg_data['trl_data'])
    tru_data = HSICD_data(data_sets, cfg_data['tru_data'])
    test_data = HSICD_data(data_sets, cfg_data['test_data'])

    # Load model
    model = DCENet(in_fea_num).to(device)
    optimizer = optim.SGD(model.parameters(), lr=cfg_optim['lr'], momentum=cfg_optim['momentum'],
                          weight_decay=cfg_optim['weight_decay'])
    # train
    # train_ULiter(trl_data, tru_data, model, optimizer, device, cfg_trainUL)

    # test
    pred_trl_label, pred_trl_acc = test(trl_data, img_gt, model, device, cfg_test)
    pred_tru_label, pred_tru_acc = test(tru_data, img_gt, model, device, cfg_test)
    pred_test_label, pred_test_acc = test(test_data, img_gt, model, device, cfg_test)

    # Post processing
    predict_label = torch.cat([pred_trl_label, pred_tru_label, pred_test_label], dim=0)
    print('pred_trl_acc {:.2f}%, pred_tru_acc {:.2f}%, pred_test_acc {:.2f}%'.format(pred_trl_acc, pred_tru_acc,
                                                                                     pred_test_acc))
    predict_img = Predict_Label2Img(predict_label, img_gt)

    conf_mat, oa, kappa_co, P, R, F1, acc = accuracy_assessment(img_gt, predict_img)
    assessment_result = [round(oa, 4) * 100, round(kappa_co, 4), round(F1, 4) * 100, round(P, 4) * 100,
                         round(R, 4) * 100]
    print('assessment_result', model_name, assessment_result)

    # Store
    save_folder = cfg_test['save_folder']
    save_name = cfg_test['save_name']
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    io.savemat(save_folder + '/' + save_name + ".mat",
               {"predict_img": np.array(predict_img.cpu()), "oa": assessment_result})
    imageio.imwrite(save_folder + '/' + save_name + '+predict_img.png', predict_img)


if __name__ == '__main__':
    main()
