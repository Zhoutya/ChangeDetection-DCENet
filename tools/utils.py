import torch
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score


def accuracy_assessment(img_gt, changed_map):
    """
    assess accuracy of changed map based on ground truth
    """
    esp = 1e-6
    height, width = changed_map.shape
    changed_map_ = np.reshape(changed_map, (-1,))
    img_gt_ = np.reshape(img_gt, (-1,))

    cm = np.ones((height * width,))
    cm[changed_map_ == 1] = 2
    cm[changed_map_ == 0] = 1

    gt = np.zeros((height * width,))
    gt[img_gt_ == 1] = 2
    gt[img_gt_ == 0] = 1

    # sklearn.metrics.confusion_matrix API
    conf_mat = confusion_matrix(y_true=gt, y_pred=cm, labels=[1, 2])
    kappa_co = cohen_kappa_score(y1=gt, y2=cm, labels=[1, 2])

    TN, FP, FN, TP = conf_mat.ravel()
    P = TP / (TP + FP + esp)
    R = TP / (TP + FN + esp)
    F1 = 2 * P * R / (P + R + esp)
    acc = (TP + TN) / (TP + TN + FP + FN + esp)
    oa = np.sum(conf_mat.diagonal()) / np.sum(conf_mat)

    return conf_mat, oa, kappa_co, P, R, F1, acc


def Predict_Label2Img(predict_label, img_gt):
    predict_img = torch.zeros_like(img_gt)
    num = predict_label.shape[0]  # 111583

    for i in range(num):
        x = int(predict_label[i][1])
        y = int(predict_label[i][2])
        l = predict_label[i][3]
        predict_img[x][y] = l

    return predict_img