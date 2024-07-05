import os, sys, math, cc3d, fastremap, csv, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage

import torch
import torch.nn.functional as F

from monai.transforms import Compose
from monai.data import decollate_batch
from monai.transforms import Invertd, SaveImaged

NUM_CLASS = 32


TEMPLATE={
    '01': [1,2,3,4,5,6,7,8,9,10,11,12,13,14],
    '01_2': [1,3,4,5,6,7,11,14],
    '02': [1,3,4,5,6,7,11,14],
    '03': [6],
    '04': [6,27], # post process
    '05': [2,3,26,32], # post process
    '06': [1,2,3,4,6,7,11,16,17],
    '07': [6,1,3,2,7,4,5,11,14,18,19,12,13,20,21,23,24],
    '08': [6, 2, 3, 1, 11],
    '09': [1,2,3,4,5,6,7,8,9,11,12,13,14,21,22],
    '12': [6,21,16,17,2,3],  
    '13': [6,2,3,1,11,8,9,7,4,5,12,13,25], 
    '14': [8,12,13,25,18,14,4,9,3,2,6,11,19,1,7],     
    '10_03': [6, 27], # post process
    '10_06': [30],
    '10_07': [11, 28], # post process
    '10_08': [15, 29], # post process
    '10_09': [1],
    '10_10': [31],
    '15': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17] ## total segmentation
}

ORGAN_NAME = ['Spleen', 'Right Kidney', 'Left Kidney', 'Gall Bladder', 'Esophagus', 
                'Liver', 'Stomach', 'Aorta', 'Postcava', 'Portal Vein and Splenic Vein',
                'Pancreas', 'Right Adrenal Gland', 'Left Adrenal Gland', 'Duodenum', 'Hepatic Vessel',
                'Right Lung', 'Left Lung', 'Colon', 'Intestine', 'Rectum', 
                'Bladder', 'Prostate', 'Left Head of Femur', 'Right Head of Femur', 'Celiac Truck',
                'Kidney Tumor', 'Liver Tumor', 'Pancreas Tumor', 'Hepatic Vessel Tumor', 'Lung Tumor', 'Colon Tumor', 'Kidney Cyst']

## mapping to original setting
MERGE_MAPPING_v1 = {
    '01': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14)],
    '02': [(1,1), (3,3), (4,4), (5,5), (6,6), (7,7), (11,11), (14,14)],
    '03': [(6,1)],
    '04': [(6,1), (27,2)],
    '05': [(2,1), (3,1), (26, 2), (32,3)],
    '06': [(1,1), (2,2), (3,3), (4,4), (6,5), (7,6), (11,7), (16,8), (17,9)],
    '07': [(1,2), (2,4), (3,3), (4,6), (5,7), (6,1), (7,5), (11,8), (12,12), (13,12), (14,9), (18,10), (19,11), (20,13), (21,14), (23,15), (24,16)],
    '08': [(1,3), (2,2), (3,2), (6,1), (11,4)],
    '09': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (11,10), (12,11), (13,12), (14,13), (21,14), (22,15)],
    '10_03': [(6,1), (27,2)],
    '10_06': [(30,1)],
    '10_07': [(11,1), (28,2)],
    '10_08': [(15,1), (29,2)],
    '10_09': [(1,1)],
    '10_10': [(31,1)],
    '12': [(2,4), (3,4), (21,2), (6,1), (16,3), (17,3)],  
    '13': [(1,3), (2,2), (3,2), (4,8), (5,9), (6,1), (7,7), (8,5), (9,6), (11,4), (12,10), (13,11), (25,12)],
    '14': [(1,18), (2,11), (3,10), (4,8), (6,12), (7,19), (8,1), (9,5), (11,13), (12,2), (13,2), (14,7), (18,6), (19,16), (25,5)],
    '15': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14), (16,16), (17,17), (18,18)],
}

## split left and right organ more than dataset defined
## expand on the original class number 
MERGE_MAPPING_v2 = {
    '01': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14)],
    '02': [(1,1), (3,3), (4,4), (5,5), (6,6), (7,7), (11,11), (14,14)],
    '03': [(6,1)],
    '04': [(6,1), (27,2)],
    '05': [(2,1), (3,3), (26, 2), (32,3)],
    '06': [(1,1), (2,2), (3,3), (4,4), (6,5), (7,6), (11,7), (16,8), (17,9)],
    '07': [(1,2), (2,4), (3,3), (4,6), (5,7), (6,1), (7,5), (11,8), (12,12), (13,17), (14,9), (18,10), (19,11), (20,13), (21,14), (23,15), (24,16)],
    '08': [(1,3), (2,2), (3,5), (6,1), (11,4)],
    '09': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (11,10), (12,11), (13,12), (14,13), (21,14), (22,15)],
    '10_03': [(6,1), (27,2)],
    '10_06': [(30,1)],
    '10_07': [(11,1), (28,2)],
    '10_08': [(15,1), (29,2)],
    '10_09': [(1,1)],
    '10_10': [(31,1)],
    '12': [(2,4), (3,5), (21,2), (6,1), (16,3), (17,6)],  
    '13': [(1,3), (2,2), (3,13), (4,8), (5,9), (6,1), (7,7), (8,5), (9,6), (11,4), (12,10), (13,11), (25,12)],
    '14': [(1,18), (2,11), (3,10), (4,8), (6,12), (7,19), (8,1), (9,5), (11,13), (12,2), (13,25), (14,7), (18,6), (19,16), (25,5)],
    '15': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6), (7,7), (8,8), (9,9), (10,10), (11,11), (12,12), (13,13), (14,14), (16,16), (17,17), (18,18)],
}

THRESHOLD_DIC = {
    'Spleen': 0.5,
    'Right Kidney': 0.5,
    'Left Kidney': 0.5,
    'Gall Bladder': 0.5,
    'Esophagus': 0.5, 
    'Liver': 0.5,
    'Stomach': 0.5,
    'Arota': 0.5, 
    'Postcava': 0.5, 
    'Portal Vein and Splenic Vein': 0.5,
    'Pancreas': 0.5, 
    'Right Adrenal Gland': 0.5, 
    'Left Adrenal Gland': 0.5, 
    'Duodenum': 0.5, 
    'Hepatic Vessel': 0.5,
    'Right Lung': 0.5, 
    'Left Lung': 0.5, 
    'Colon': 0.5, 
    'Intestine': 0.5, 
    'Rectum': 0.5, 
    'Bladder': 0.5, 
    'Prostate': 0.5, 
    'Left Head of Femur': 0.5, 
    'Right Head of Femur': 0.5, 
    'Celiac Truck': 0.5,
    'Kidney Tumor': 0.5, 
    'Liver Tumor': 0.5, 
    'Pancreas Tumor': 0.5, 
    'Hepatic Vessel Tumor': 0.5, 
    'Lung Tumor': 0.5, 
    'Colon Tumor': 0.5, 
    'Kidney Cyst': 0.5
}



TUMOR_ORGAN = {
    'Kidney Tumor': [2,3], 
    'Liver Tumor': [6], 
    'Pancreas Tumor': [11], 
    'Hepatic Vessel Tumor': [15], 
    'Lung Tumor': [16,17], 
    'Colon Tumor': [18], 
    'Kidney Cyst': [2,3]
}

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    distributed = torch.distributed.is_available(
    ) and torch.distributed.is_initialized()
    if not distributed or (distributed and torch.distributed.get_rank() == 0):
        torch.save(state, filename)
        print("=> saved checkpoint '{}' (epoch {})".format(
            filename, state['epoch']))


def organ_post_process(pred_mask, organ_list, save_dir, args):
    post_pred_mask = np.zeros(pred_mask.shape)
    plot_save_path = save_dir
    log_path = args.log_name
    dataset_id = save_dir.split('/')[-2]
    case_id = save_dir.split('/')[-1]
    if not os.path.isdir(plot_save_path):
        os.makedirs(plot_save_path)
    for b in range(pred_mask.shape[0]):
        for organ in organ_list:
            if organ == 11: # both process pancreas and Portal vein and splenic vein
                post_pred_mask[b,10] = extract_topk_largest_candidates(pred_mask[b,10], 1) # for pancreas
                if 10 in organ_list:
                    post_pred_mask[b,9] = PSVein_post_process(pred_mask[b,9], post_pred_mask[b,10])
            elif organ == 16:
                try:
                    left_lung_mask, right_lung_mask = lung_post_process(pred_mask[b])
                    post_pred_mask[b,16] = left_lung_mask
                    post_pred_mask[b,15] = right_lung_mask
                except IndexError:
                    print('this case does not have lungs!')
                    shape_temp = post_pred_mask[b,16].shape
                    post_pred_mask[b,16] = np.zeros(shape_temp)
                    post_pred_mask[b,15] = np.zeros(shape_temp)
                    with open(log_path + '/' + dataset_id +'/anomaly.csv','a',newline='') as f:
                        writer = csv.writer(f)
                        content = case_id
                        writer.writerow([content])

                right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))
                
                print('left lung size: '+str(left_lung_size))
                print('right lung size: '+str(right_lung_size))

                #knn_model = KNN(n_neighbors=5,contamination=0.00001)
                right_lung_save_path = plot_save_path+'/right_lung.png'
                left_lung_save_path = plot_save_path+'/left_lung.png'
                total_anomly_slice_number=0

                if right_lung_size>left_lung_size:
                    if right_lung_size/left_lung_size > 4:
                        mid_point = int(right_lung_mask.shape[0]/2)
                        left_region = np.sum(right_lung_mask[:mid_point,:,:],axis=(0,1,2))
                        right_region = np.sum(right_lung_mask[mid_point:,:,:],axis=(0,1,2))
                        
                        if (right_region+1)/(left_region+1)>4:
                            print('this case only has right lung')
                            post_pred_mask[b,15] = right_lung_mask
                            post_pred_mask[b,16] = np.zeros(right_lung_mask.shape)
                        elif (left_region+1)/(right_region+1)>4:
                            print('this case only has left lung')
                            post_pred_mask[b,16] = right_lung_mask
                            post_pred_mask[b,15] = np.zeros(right_lung_mask.shape)
                        else:
                            print('need anomly detection')
                            print('start anomly detection at right lung')
                            try:
                                left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                    pred_mask,post_pred_mask[b,15],right_lung_save_path,b,total_anomly_slice_number)
                                post_pred_mask[b,16] = left_lung_mask
                                post_pred_mask[b,15] = right_lung_mask
                                right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                                left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))
                                while right_lung_size/left_lung_size>4 or left_lung_size/right_lung_size>4:
                                    print('still need anomly detection')
                                    if right_lung_size>left_lung_size:
                                        left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                        pred_mask,post_pred_mask[b,15],right_lung_save_path,b,total_anomly_slice_number)
                                    else:
                                        left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                        pred_mask,post_pred_mask[b,16],right_lung_save_path,b,total_anomly_slice_number)
                                    post_pred_mask[b,16] = left_lung_mask
                                    post_pred_mask[b,15] = right_lung_mask
                                    right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                                    left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))
                                print('lung seperation complete')
                            except IndexError:
                                left_lung_mask, right_lung_mask = lung_post_process(pred_mask[b])
                                post_pred_mask[b,16] = left_lung_mask
                                post_pred_mask[b,15] = right_lung_mask
                                print("cannot seperate two lungs, writing csv")
                                with open(log_path + '/' + dataset_id +'/anomaly.csv','a',newline='') as f:
                                    writer = csv.writer(f)
                                    content = case_id
                                    writer.writerow([content])
                else:
                    if left_lung_size/right_lung_size > 4:
                        mid_point = int(left_lung_mask.shape[0]/2)
                        left_region = np.sum(left_lung_mask[:mid_point,:,:],axis=(0,1,2))
                        right_region = np.sum(left_lung_mask[mid_point:,:,:],axis=(0,1,2))
                        if (right_region+1)/(left_region+1)>4:
                            print('this case only has right lung')
                            post_pred_mask[b,15] = left_lung_mask
                            post_pred_mask[b,16] = np.zeros(left_lung_mask.shape)
                        elif (left_region+1)/(right_region+1)>4:
                            print('this case only has left lung')
                            post_pred_mask[b,16] = left_lung_mask
                            post_pred_mask[b,15] = np.zeros(left_lung_mask.shape)
                        else:

                            print('need anomly detection')
                            print('start anomly detection at left lung')
                            try:
                                left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                    pred_mask,post_pred_mask[b,16],left_lung_save_path,b,total_anomly_slice_number)
                                post_pred_mask[b,16] = left_lung_mask
                                post_pred_mask[b,15] = right_lung_mask
                                right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                                left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))
                                while right_lung_size/left_lung_size>4 or left_lung_size/right_lung_size>4:
                                    print('still need anomly detection')
                                    if right_lung_size>left_lung_size:
                                        left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                        pred_mask,post_pred_mask[b,15],right_lung_save_path,b,total_anomly_slice_number)
                                    else:
                                        left_lung_mask,right_lung_mask,total_anomly_slice_number = anomly_detection(
                                        pred_mask,post_pred_mask[b,16],right_lung_save_path,b,total_anomly_slice_number)
                                    post_pred_mask[b,16] = left_lung_mask
                                    post_pred_mask[b,15] = right_lung_mask
                                    right_lung_size = np.sum(post_pred_mask[b,15],axis=(0,1,2))
                                    left_lung_size = np.sum(post_pred_mask[b,16],axis=(0,1,2))

                                print('lung seperation complete')
                            except IndexError:
                                left_lung_mask, right_lung_mask = lung_post_process(pred_mask[b])
                                post_pred_mask[b,16] = left_lung_mask
                                post_pred_mask[b,15] = right_lung_mask
                                print("cannot seperate two lungs, writing csv")
                                with open(log_path + '/' + dataset_id +'/anomaly.csv','a',newline='') as f:
                                    writer = csv.writer(f)
                                    content = case_id
                                    writer.writerow([content])
                print('find number of anomaly slice: '+str(total_anomly_slice_number))
            elif organ == 17:
                continue ## the le
            elif organ in [1,2,3,4,5,6,7,8,9,12,13,14,18,19,20,21,22,23,24,25]: ## rest organ index
                post_pred_mask[b,organ-1] = extract_topk_largest_candidates(pred_mask[b,organ-1], 1)
            # elif organ in [28,29,30,31,32]:
            #     post_pred_mask[b,organ-1] = extract_topk_largest_candidates(pred_mask[b,organ-1], TUMOR_NUM[ORGAN_NAME[organ-1]], area_least=TUMOR_SIZE[ORGAN_NAME[organ-1]])
            elif organ in [26,27]:
                organ_mask = merge_and_top_organ(pred_mask[b], TUMOR_ORGAN[ORGAN_NAME[organ-1]])
                post_pred_mask[b,organ-1] = organ_region_filter_out(pred_mask[b,organ-1], organ_mask)
                # post_pred_mask[b,organ-1] = extract_topk_largest_candidates(post_pred_mask[b,organ-1], TUMOR_NUM[ORGAN_NAME[organ-1]], area_least=TUMOR_SIZE[ORGAN_NAME[organ-1]])
            else:
                post_pred_mask[b,organ-1] = pred_mask[b,organ-1]
    return post_pred_mask

def lung_overlap_post_process(pred_mask):
    new_mask = np.zeros(pred_mask.shape, np.uint8)
    new_mask[pred_mask==1] = 1
    label_out = cc3d.connected_components(new_mask, connectivity=26)

    areas = {}
    for label, extracted in cc3d.each(label_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)
    num_candidates = len(candidates)
    if num_candidates!=1:
        print('start separating two lungs!')
        ONE = int(candidates[0][0])
        TWO = int(candidates[1][0])


        print('number of connected components:'+str(len(candidates)))
        a1,b1,c1 = np.where(label_out==ONE)
        a2,b2,c2 = np.where(label_out==TWO)
        
        left_lung_mask = np.zeros(label_out.shape)
        right_lung_mask = np.zeros(label_out.shape)

        if np.mean(a1) < np.mean(a2):
            left_lung_mask[label_out==ONE] = 1
            right_lung_mask[label_out==TWO] = 1
        else:
            right_lung_mask[label_out==ONE] = 1
            left_lung_mask[label_out==TWO] = 1
        erosion_left_lung_size = np.sum(left_lung_mask,axis=(0,1,2))
        erosion_right_lung_size = np.sum(right_lung_mask,axis=(0,1,2))
        print('erosion left lung size:'+str(erosion_left_lung_size))
        print('erosion right lung size:'+ str(erosion_right_lung_size))
        return num_candidates,left_lung_mask, right_lung_mask
    else:
        print('current iteration cannot separate lungs, erosion iteration + 1')
        ONE = int(candidates[0][0])
        print('number of connected components:'+str(len(candidates)))
        lung_mask = np.zeros(label_out.shape)
        lung_mask[label_out == ONE]=1
        lung_overlapped_mask_size = np.sum(lung_mask,axis=(0,1,2))
        print('lung overlapped mask size:' + str(lung_overlapped_mask_size))

        return num_candidates,lung_mask

def find_best_iter_and_masks(lung_mask):
    iter=1
    print('current iteration:' + str(iter))
    struct2 = ndimage.generate_binary_structure(3, 3)
    erosion_mask= ndimage.binary_erosion(lung_mask, structure=struct2,iterations=iter)
    candidates_and_masks = lung_overlap_post_process(erosion_mask)
    while candidates_and_masks[0]==1:
        iter +=1
        print('current iteration:' + str(iter))
        erosion_mask= ndimage.binary_erosion(lung_mask, structure=struct2,iterations=iter)
        candidates_and_masks = lung_overlap_post_process(erosion_mask)
    print('check if components are valid')
    left_lung_erosion_mask = candidates_and_masks[1]
    right_lung_erosion_mask = candidates_and_masks[2]
    left_lung_erosion_mask_size = np.sum(left_lung_erosion_mask,axis = (0,1,2))
    right_lung_erosion_mask_size = np.sum(right_lung_erosion_mask,axis = (0,1,2))
    while left_lung_erosion_mask_size/right_lung_erosion_mask_size>4 or right_lung_erosion_mask_size/left_lung_erosion_mask_size>4:
        print('components still have large difference, erosion interation + 1')
        iter +=1
        print('current iteration:' + str(iter))
        erosion_mask= ndimage.binary_erosion(lung_mask, structure=struct2,iterations=iter)
        candidates_and_masks = lung_overlap_post_process(erosion_mask)
        while candidates_and_masks[0]==1:
            iter +=1
            print('current iteration:' + str(iter))
            erosion_mask= ndimage.binary_erosion(lung_mask, structure=struct2,iterations=iter)
            candidates_and_masks = lung_overlap_post_process(erosion_mask)
        left_lung_erosion_mask = candidates_and_masks[1]
        right_lung_erosion_mask = candidates_and_masks[2]
        left_lung_erosion_mask_size = np.sum(left_lung_erosion_mask,axis = (0,1,2))
        right_lung_erosion_mask_size = np.sum(right_lung_erosion_mask,axis = (0,1,2))
    print('erosion done, best iteration: '+str(iter))



    print('start dilation')
    left_lung_erosion_mask = candidates_and_masks[1]
    right_lung_erosion_mask = candidates_and_masks[2]

    erosion_part_mask = lung_mask - left_lung_erosion_mask - right_lung_erosion_mask
    left_lung_dist = np.ones(left_lung_erosion_mask.shape)
    right_lung_dist = np.ones(right_lung_erosion_mask.shape)
    left_lung_dist[left_lung_erosion_mask==1]=0
    right_lung_dist[right_lung_erosion_mask==1]=0
    left_lung_dist_map = ndimage.distance_transform_edt(left_lung_dist)
    right_lung_dist_map = ndimage.distance_transform_edt(right_lung_dist)
    left_lung_dist_map[erosion_part_mask==0]=0
    right_lung_dist_map[erosion_part_mask==0]=0
    left_lung_adding_map = left_lung_dist_map < right_lung_dist_map
    right_lung_adding_map = right_lung_dist_map < left_lung_dist_map
    
    left_lung_erosion_mask[left_lung_adding_map==1]=1
    right_lung_erosion_mask[right_lung_adding_map==1]=1

    left_lung_mask = left_lung_erosion_mask
    right_lung_mask = right_lung_erosion_mask
    # left_lung_mask = ndimage.binary_dilation(left_lung_erosion_mask, structure=struct2,iterations=iter)
    # right_lung_mask = ndimage.binary_dilation(right_lung_erosion_mask, structure=struct2,iterations=iter)
    print('dilation complete')
    left_lung_mask_fill_hole = ndimage.binary_fill_holes(left_lung_mask)
    right_lung_mask_fill_hole = ndimage.binary_fill_holes(right_lung_mask)
    left_lung_size = np.sum(left_lung_mask_fill_hole,axis=(0,1,2))
    right_lung_size = np.sum(right_lung_mask_fill_hole,axis=(0,1,2))
    print('new left lung size:'+str(left_lung_size))
    print('new right lung size:' + str(right_lung_size))
    return left_lung_mask_fill_hole,right_lung_mask_fill_hole

def anomly_detection(pred_mask, post_pred_mask, save_path, batch, anomly_num):
    total_anomly_slice_number = anomly_num
    df = get_dataframe(post_pred_mask)
    # lung_pred_df = fit_model(model,lung_df)
    lung_df = df[df['array_sum']!=0]
    lung_df['SMA20'] = lung_df['array_sum'].rolling(20,min_periods=1,center=True).mean()
    lung_df['STD20'] = lung_df['array_sum'].rolling(20,min_periods=1,center=True).std()
    lung_df['SMA7'] = lung_df['array_sum'].rolling(7,min_periods=1,center=True).mean()
    lung_df['upper_bound'] = lung_df['SMA20']+2*lung_df['STD20']
    lung_df['Predictions'] = lung_df['array_sum']>lung_df['upper_bound']
    lung_df['Predictions'] = lung_df['Predictions'].astype(int)
    lung_df.dropna(inplace=True)
    anomly_df = lung_df[lung_df['Predictions']==1]
    anomly_slice = anomly_df['slice_index'].to_numpy()
    anomly_value = anomly_df['array_sum'].to_numpy()
    anomly_SMA7 = anomly_df['SMA7'].to_numpy()

    print('decision made')
    if len(anomly_df)!=0:
        print('anomaly point detected')
        print('check if the anomaly points are real')
        real_anomly_slice = []
        for i in range(len(anomly_df)):
            if anomly_value[i] > anomly_SMA7[i]+200:
                print('the anomaly point is real')
                real_anomly_slice.append(anomly_slice[i])
                total_anomly_slice_number+=1
        
        if len(real_anomly_slice)!=0:

            
            plot_anomalies(lung_df,save_dir=save_path)
            print('anomaly detection plot created')
            for s in real_anomly_slice:
                pred_mask[batch,15,:,:,s]=0
                pred_mask[batch,16,:,:,s]=0
            left_lung_mask, right_lung_mask = lung_post_process(pred_mask[batch])
            left_lung_size = np.sum(left_lung_mask,axis=(0,1,2))
            right_lung_size = np.sum(right_lung_mask,axis=(0,1,2))
            print('new left lung size:'+str(left_lung_size))
            print('new right lung size:' + str(right_lung_size))
            return left_lung_mask,right_lung_mask,total_anomly_slice_number
        else: 
            print('the anomaly point is not real, start separate overlapping')
            left_lung_mask,right_lung_mask = find_best_iter_and_masks(post_pred_mask)
            return left_lung_mask,right_lung_mask,total_anomly_slice_number


    print('overlap detected, start erosion and dilation')
    left_lung_mask,right_lung_mask = find_best_iter_and_masks(post_pred_mask)

    return left_lung_mask,right_lung_mask,total_anomly_slice_number

def get_dataframe(post_pred_mask):
    target_array = post_pred_mask
    target_array_sum = np.sum(target_array,axis=(0,1))
    slice_index = np.arange(target_array.shape[-1])
    df = pd.DataFrame({'slice_index':slice_index,'array_sum':target_array_sum})
    return df

def plot_anomalies(df, x='slice_index', y='array_sum',save_dir=None):
    # categories will be having values from 0 to n
    # for each values in 0 to n it is mapped in colormap
    categories = df['Predictions'].to_numpy()
    colormap = np.array(['g', 'r'])

    f = plt.figure(figsize=(12, 4))
    f = plt.plot(df[x],df['SMA20'],'b')
    f = plt.plot(df[x],df['upper_bound'],'y')
    f = plt.scatter(df[x], df[y], c=colormap[categories],alpha=0.3)
    f = plt.xlabel(x)
    f = plt.ylabel(y)
    plt.legend(['Simple moving average','upper bound','predictions'])
    if save_dir is not None:
        plt.savefig(save_dir)
    plt.clf()

def merge_and_top_organ(pred_mask, organ_list):
    ## merge 
    out_mask = np.zeros(pred_mask.shape[1:], np.uint8)
    for organ in organ_list:
        out_mask = np.logical_or(out_mask, pred_mask[organ-1])
    ## select the top k, for righr left case
    out_mask = extract_topk_largest_candidates(out_mask, len(organ_list))

    return out_mask

def organ_region_filter_out(tumor_mask, organ_mask):
    ## dialtion
    organ_mask = ndimage.binary_closing(organ_mask, structure=np.ones((5,5,5)))
    organ_mask = ndimage.binary_dilation(organ_mask, structure=np.ones((5,5,5)))
    ## filter out
    tumor_mask = organ_mask * tumor_mask

    return tumor_mask


def PSVein_post_process(PSVein_mask, pancreas_mask):
    xy_sum_pancreas = pancreas_mask.sum(axis=0).sum(axis=0)
    z_non_zero = np.nonzero(xy_sum_pancreas)
    z_value = np.min(z_non_zero) ## the down side of pancreas
    new_PSVein = PSVein_mask.copy()
    new_PSVein[:,:,:z_value] = 0
    return new_PSVein

def lung_post_process(pred_mask):
    new_mask = np.zeros(pred_mask.shape[1:], np.uint8)
    new_mask[pred_mask[15] == 1] = 1
    new_mask[pred_mask[16] == 1] = 1
    label_out = cc3d.connected_components(new_mask, connectivity=26)
    
    areas = {}
    for label, extracted in cc3d.each(label_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    ONE = int(candidates[0][0])
    TWO = int(candidates[1][0])
    
    a1,b1,c1 = np.where(label_out==ONE)
    a2,b2,c2 = np.where(label_out==TWO)
    
    left_lung_mask = np.zeros(label_out.shape)
    right_lung_mask = np.zeros(label_out.shape)

    if np.mean(a1) < np.mean(a2):
        left_lung_mask[label_out==ONE] = 1
        right_lung_mask[label_out==TWO] = 1
    else:
        right_lung_mask[label_out==ONE] = 1
        left_lung_mask[label_out==TWO] = 1
    
    return left_lung_mask, right_lung_mask

def extract_topk_largest_candidates(npy_mask, organ_num, area_least=0):
    ## npy_mask: w, h, d
    ## organ_num: the maximum number of connected component
    out_mask = np.zeros(npy_mask.shape, np.uint8)
    t_mask = npy_mask.copy()
    keep_topk_largest_connected_object(t_mask, organ_num, area_least, out_mask, 1)

    return out_mask


def keep_topk_largest_connected_object(npy_mask, k, area_least, out_mask, out_label):
    labels_out = cc3d.connected_components(npy_mask, connectivity=26)
    areas = {}
    for label, extracted in cc3d.each(labels_out, binary=True, in_place=True):
        areas[label] = fastremap.foreground(extracted)
    candidates = sorted(areas.items(), key=lambda item: item[1], reverse=True)

    for i in range(min(k, len(candidates))):
        if candidates[i][1] > area_least:
            out_mask[labels_out == int(candidates[i][0])] = out_label

def threshold_organ(data, organ=None, threshold=None):
    ### threshold the sigmoid value to hard label
    ## data: sigmoid value
    ## threshold_list: a list of organ threshold
    B = data.shape[0]
    threshold_list = []
    if organ:
        THRESHOLD_DIC[organ] = threshold
    for key, value in THRESHOLD_DIC.items():
        threshold_list.append(value)
    threshold_list = torch.tensor(threshold_list).repeat(B, 1).reshape(B,len(threshold_list),1,1,1).cuda()
    pred_hard = data > threshold_list
    return pred_hard


def visualize_label(batch, save_dir, input_transform):
    ### function: save the prediction result into dir
    ## Input
    ## batch: the batch dict output from the monai dataloader
    ## one_channel_label: the predicted reuslt with same shape as label
    ## save_dir: the directory for saving
    ## input_transform: the dataloader transform
    post_transforms = Compose([
        Invertd(
            keys=["label", 'one_channel_label_v1', 'one_channel_label_v2'], #, 'split_label'
            transform=input_transform,
            orig_keys="image",
            nearest_interp=True,
            to_tensor=True,
        ),
        SaveImaged(keys="label", 
                meta_keys="label_meta_dict" , 
                output_dir=save_dir, 
                output_postfix="gt", 
                resample=False
        ),
        SaveImaged(keys='one_channel_label_v1', 
                meta_keys="label_meta_dict" , 
                output_dir=save_dir, 
                output_postfix="result_v1", 
                resample=False
        ),
        SaveImaged(keys='one_channel_label_v2', 
                meta_keys="label_meta_dict" , 
                output_dir=save_dir, 
                output_postfix="result_v2", 
                resample=False
        ),
    ])
    
    batch = [post_transforms(i) for i in decollate_batch(batch)]


def merge_label(pred_bmask, name):
    B, C, W, H, D = pred_bmask.shape
    merged_label_v1 = torch.zeros(B,1,W,H,D).cuda()
    merged_label_v2 = torch.zeros(B,1,W,H,D).cuda()
    for b in range(B):
        template_key = get_key(name[b])
        transfer_mapping_v1 = MERGE_MAPPING_v1[template_key]
        transfer_mapping_v2 = MERGE_MAPPING_v2[template_key]
        organ_index = []
        for item in transfer_mapping_v1:
            src, tgt = item
            merged_label_v1[b][0][pred_bmask[b][src-1]==1] = tgt
        for item in transfer_mapping_v2:
            src, tgt = item
            merged_label_v2[b][0][pred_bmask[b][src-1]==1] = tgt
            # organ_index.append(src-1)
        # organ_index = torch.tensor(organ_index).cuda()
        # predicted_prob = pred_sigmoid[b][organ_index]
    return merged_label_v1, merged_label_v2


def get_key(name):
    ## input: name
    ## output: the corresponding key
    dataset_index = int(name[0:2])
    if dataset_index == 10:
        template_key = name[0:2] + '_' + name[17:19]
    else:
        template_key = name[0:2]
    return template_key

def dice_score(preds, labels, spe_sen=False):
    assert preds.shape[0] == labels.shape[0], "predict & target batch size don't match"
    preds = torch.where(preds > 0.5, 1., 0.)
    predict = preds.contiguous().view(1, -1)
    target = labels.contiguous().view(1, -1)

    tp = torch.sum(torch.mul(predict, target))
    fn = torch.sum(torch.mul(predict!=1, target))
    fp = torch.sum(torch.mul(predict, target!=1))
    tn = torch.sum(torch.mul(predict!=1, target!=1))

    den = torch.sum(predict) + torch.sum(target) + 1

    dice = 2 * tp / den
    recall = tp/(tp+fn)
    precision = tp/(tp+fp)
    specificity = tn/(fp + tn)

    if spe_sen:
        return dice, recall, precision, specificity
    else:
        return dice, recall, precision

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class WindowAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, k=250, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.k = k
        self.reset()

    def reset(self):
        from collections import deque
        self.vals = deque(maxlen=self.k)
        self.counts = deque(maxlen=self.k)
        self.val = 0
        self.avg = 0

    def update(self, val, n=1):
        self.vals.append(val)
        self.counts.append(n)
        self.val = val
        self.avg = sum([v * c for v, c in zip(self.vals, self.counts)]) / sum(
            self.counts)

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", tbwriter=None, rank=0):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.tbwriter = tbwriter
        self.rank = rank

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        if self.rank == 0:
            print('\t'.join(entries))
            sys.stdout.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

    def tbwrite(self, batch):
        if self.tbwriter is None:
            return
        scalar_dict = self.tb_scalar_dict()
        for k, v in scalar_dict.items():
            self.tbwriter.add_scalar(k, v, batch)

    def tb_scalar_dict(self):
        out = {}
        for meter in self.meters:
            val = meter.avg
            tag = meter.name
            sclrval = val
            out[tag] = sclrval
        return out

class CheckpointManager:
    def __init__(self,
                 modules,
                 ckpt_dir,
                 epoch_size,
                 epochs,
                 save_freq=None,
                 save_freq_mints=None):
        self.modules = modules
        self.ckpt_dir = ckpt_dir
        self.epoch_size = epoch_size
        self.epochs = epochs
        self.save_freq = save_freq
        self.save_freq_mints = save_freq_mints
        self.retain_num_ckpt = 0

        self.time = time.time()
        self.distributed = torch.distributed.is_available(
        ) and torch.distributed.is_initialized()
        self.world_size = torch.distributed.get_world_size(
        ) if self.distributed else 1
        self.rank = torch.distributed.get_rank() if self.distributed else 0

        os.makedirs(os.path.join(self.ckpt_dir), exist_ok=True)

    def resume(self):
        ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')
        start_epoch = 0
        if os.path.isfile(ckpt_fname):
            checkpoint = torch.load(ckpt_fname, map_location='cpu')

            # Load state dict
            for k in self.modules:
                self.modules[k].load_state_dict(checkpoint[k])
            start_epoch = checkpoint['epoch']
            print("=> loaded checkpoint '{}' (epoch {})".format(
                ckpt_fname, checkpoint['epoch']))
        return start_epoch

    def timed_checkpoint(self, save_dict=None):
        t = time.time() - self.time
        t_all = [t for _ in range(self.world_size)]
        if self.world_size > 1:
            torch.distributed.all_gather_object(t_all, t)
        if min(t_all) > self.save_freq_mints * 60:
            self.time = time.time()
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_latest.pth')

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)

    def midway_epoch_checkpoint(self, epoch, batch_i, save_dict=None):
        # if ((batch_i + 1) / float(self.epoch_size) % self.save_freq) < (
                # batch_i / float(self.epoch_size) % self.save_freq):
        if batch_i % self.save_freq==0:
            ckpt_fname = os.path.join(self.ckpt_dir,
                                      'checkpoint_{:010.4f}.pth')
            ckpt_fname = ckpt_fname.format(epoch +
                                           batch_i / float(self.epoch_size))

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)
                ckpt_fname = os.path.join(self.ckpt_dir,
                                          'checkpoint_latest.pth')
                save_checkpoint(state, is_best=False, filename=ckpt_fname)

    def end_epoch_checkpoint(self, epoch, save_dict=None):
        if (epoch % self.save_freq
                == 0) or self.save_freq < 1 or epoch == self.epochs:
            ckpt_fname = os.path.join(self.ckpt_dir, 'checkpoint_{:04d}.pth')
            ckpt_fname = ckpt_fname.format(epoch)

            state = self.create_state_dict(save_dict)
            if self.rank == 0:
                save_checkpoint(state, is_best=False, filename=ckpt_fname)
                ckpt_fname = os.path.join(self.ckpt_dir,
                                          'checkpoint_latest.pth')
                save_checkpoint(state, is_best=False, filename=ckpt_fname)

            if self.retain_num_ckpt > 0:
                ckpt_fname = os.path.join(self.ckpt_dir,
                                          'checkpoint_{:04d}.pth')
                ckpt_fname = ckpt_fname.format(epoch - self.save_freq *
                                               (self.retain_num_ckpt + 1))
                if os.path.exists(ckpt_fname):
                    os.remove(ckpt_fname.format(ckpt_fname))

    def create_state_dict(self, save_dict):
        state = {k: self.modules[k].state_dict() for k in self.modules}
        if save_dict is not None:
            state.update(save_dict)
        return state

    def checkpoint(self, epoch, batch_i=None, save_dict=None):
        if batch_i is None:
            self.end_epoch_checkpoint(epoch, save_dict)
        else:
            if batch_i % 100 == 0:
                self.timed_checkpoint(save_dict)
            self.midway_epoch_checkpoint(epoch, batch_i, save_dict=save_dict)
                
def adjust_learning_rate(optimizer, epoch, args, epoch_size=None):
    """Decay the learning rate based on schedule"""
    init_lr = args.lr
    if args.lr_schedule == 'constant':
        cur_lr = init_lr

    elif args.lr_schedule == 'cos':
        cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.max_epoch))

    elif args.lr_schedule == 'triangle':
        T = args.lr_schedule_period
        t = (epoch * epoch_size) % T
        if t < T / 2:
            cur_lr = args.lr + t / (T / 2.) * (args.max_lr - args.lr)
        else:
            cur_lr = args.lr + (T-t) / (T / 2.) * (args.max_lr - args.lr)

    else:
        raise ValueError('LR schedule unknown.')

    if args.exit_decay > 0:
        start_decay_epoch = args.max_epoch * (1. - args.exit_decay)
        if epoch > start_decay_epoch:
            mult = 0.5 * (1. + math.cos(math.pi * (epoch - start_decay_epoch) / (args.max_epoch - start_decay_epoch)))
            cur_lr = cur_lr * mult

    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr
    return cur_lr

def seconds_to_hms(seconds):
    days, remainder = divmod(seconds, 86400)  # 1 day = 24 * 60 * 60 seconds
    hours, remainder = divmod(remainder, 3600)  # 1 hour = 60 * 60 seconds
    minutes, seconds = divmod(remainder, 60)  # 1 minute = 60 seconds
    return int(days), int(hours), int(minutes), int(seconds)

def calculate_remaining_time(start_time, current_step, total_steps):
    elapsed_time = time.time() - start_time
    time_per_step = elapsed_time / current_step
    remaining_steps = total_steps - current_step
    remaining_time = remaining_steps * time_per_step
    days, hours, minutes, seconds = seconds_to_hms(remaining_time)
    return days, hours, minutes, seconds