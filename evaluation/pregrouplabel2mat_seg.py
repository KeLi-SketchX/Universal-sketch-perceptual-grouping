import h5py
import numpy as np
import os
import scipy.io as sci
import pdb

def line_label2group_label(line_labels,str_labels_matrix):
    str_labels = str_labels_matrix[1,:]
    temp_i =1
    while np.sum(str_labels)==0:
        temp_i +=1
        str_labels= str_labels_matrix[temp_i,:]
    line_group_label = []
    gap_idx = np.where(str_labels==0)[0]
    str_num = len(gap_idx)
    line_idx = 0
    for str_idx,str_label in enumerate(str_labels):
        if str_label==0:
            line_label = line_labels[line_idx]
            line_group_label.append(line_label)
        else:
            line_label = line_labels[line_idx]
            line_group_label.append(line_label)
            line_idx+=1
    stroke_group_label = np.zeros((str_num,1))
    for i in range(len(gap_idx)):
        if i==len(gap_idx)-1:
            b_off = gap_idx[i]
            e_off = len(line_group_label)-1
        else:
            b_off = gap_idx[i]
            e_off = gap_idx[i+1]
        temp_line_label = line_group_label[b_off:e_off]
        unique_labels = np.unique(temp_line_label)
        if len(unique_labels)==1:
            stroke_group_label[i,0]=unique_labels[0]
        else:
            len_uni_label = np.zeros((len(unique_labels),1))
            for unique_label_idx, unique_label in enumerate(unique_labels):
                len_uni_label[unique_label_idx,0] = len(np.where(line_group_label==unique_label)[0])
            max_len = np.max(len_uni_label)
            max_idx = np.where(len_uni_label==max_len)[0]
            stroke_group_label[i, 0] = unique_labels[max_idx[0]]

    return stroke_group_label



pre_label_file = '/import/vision-datasets/kl303/PycharmProjects/ECCV_segmentaion/8563/pre_group_labels.h5'
str_label_file = '/import/vision-datasets/kl303/PycharmProjects/ECCV_segmentaion/8563/str_labels.h5'
out_put_file = '/import/vision-datasets/kl303/PycharmProjects/BSR/bench/pre_label/segmentation/'
pre_label_f = h5py.File(pre_label_file,'r')
str_label_f = h5py.File(str_label_file,'r')

test_file_name = '/import/vision-datasets/kl303/PycharmProjects/ECCV_segmentaion/name_list.npy'
# test_f=open(test_file_name,'r')
name_list = np.load(test_file_name)

for key_idx in range(len(pre_label_f.keys())):
    key = str(key_idx)
    pre_label = pre_label_f[key].value
    str_label_matrix = str_label_f[key].value
    stroke_group_label = line_label2group_label(pre_label,str_label_matrix)


    line=name_list[key_idx]
    mat_file_name = out_put_file+line[:-4]+'.mat'
    sci.savemat(mat_file_name,{'label':stroke_group_label})

str_label_f.close()
pre_label_f.close()cd