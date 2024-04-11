'''
 Dataset Source:
    Farmland: http://crabwq.github.io/
    Hermiston: https://citius.usc.es/investigacion/datasets/hyperspectral-change-detection-dataset
'''
import numpy as np
from scipy.io import loadmat

def get_Hermiston_dataset():
    data_set_before = loadmat(r'../../datasets/Hermiston/USA_Change_Dataset.mat')['T1']
    data_set_after = loadmat(r'../../datasets/Hermiston/USA_Change_Dataset.mat')['T2']
    ground_truth = loadmat(r'../../datasets/Hermiston/USA_Change_Dataset.mat')['Binary']

    img1 = data_set_before.astype('float32')  # (420, 140, 154)
    img2 = data_set_after.astype('float32')  # (420, 140, 154)
    gt = ground_truth.astype('float32')  # (420, 140)

    return img1, img2, gt


def get_dataset(current_dataset):
    if current_dataset == 'Hermiston':
        return get_Hermiston_dataset()  # Hermiston(307, 241, 154), gt[0. 1.]
