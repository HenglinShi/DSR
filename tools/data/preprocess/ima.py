import os
from os.path import join
import sys
import json
import numpy as np
from .read_openpose import read_openpose
import pandas as pd
import os.path as osp
import pdb


def ima_extract(dataset_path, openpose_path, out_path):

    video_names_train = ['video_000135', 'video_000099', 'video_000297', 'video_000088', 'video_000268', 
                        'video_000295', 'video_000296', 'video_000369', 'video_000282', 'video_000347', 
                        'video_000191', 'video_000399', 'video_000011', 'video_000398', 'video_000068', 
                        'video_000385', 'video_000021', 'video_000001', 'video_000267', 'video_000290', 
                        'video_000005', 'video_000052', 'video_000073', 'video_000389', 'video_000173', 
                        'video_000352', 'video_000283', 'video_000285', 'video_000110', 'video_000346', 
                        'video_000127', 'video_000031', 'video_000344', 'video_000169', 'video_000292', 
                        'video_000339', 'video_000349', 'video_000190', 'video_000409', 'video_000364', 
                        'video_000379', 'video_000047', 'video_000338', 'video_000112', 'video_000236', 
                        'video_000204', 'video_000269', 'video_000287', 'video_000141', 'video_000291', 
                        'video_000123', 'video_000360', 'video_000227', 'video_000244', 'video_000358', 
                        'video_000353', 'video_000288', 'video_000340', 'video_000004']



    video_names_test = ['video_000106', 'video_000284', 'video_000345', 'video_000070', 'video_000079', 
                        'video_000107', 'video_000122', 'video_000121', 'video_000241', 'video_000059', 
                        'video_000000', 'video_000394', 'video_000405', 'video_000341', 'video_000111', 
                        'video_000186', 'video_000179', 'video_000412', 'video_000090', 'video_000086', 
                        'video_000286', 'video_000172', 'video_000077', 'video_000276', 'video_000396', 
                        'video_000348']

    video_sizes_df = pd.read_excel('/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/annotations/annotations_size.xlsx')
    video_sizes_dict = {}
    n_row, _ = video_sizes_df.shape
    for row in range(n_row):

        if pd.isna(video_sizes_df['Unnamed: 7'][row]) is True:
            height = -1
            width = -1
        elif video_sizes_df['Unnamed: 7'][row] == 'resize' :
            height = video_sizes_df.height_annotated[row]
            width = video_sizes_df.width_annotated[row]
            pass
        elif video_sizes_df['Unnamed: 7'][row] == 'original' or video_sizes_df['Unnamed: 7'][row] == 'original/resize': 
            height = video_sizes_df.height_real[row]
            width = video_sizes_df.width_real[row]
            pass

        elif video_sizes_df['Unnamed: 7'][row] == 'Neither': 
            height = video_sizes_df.height_real[row]
            width = video_sizes_df.width_real[row]
            pass
        else:
            raise NotImplementedError()
        video_sizes_dict[video_sizes_df.video_name[row]] = [width, height]


    body_parts = [
        'LHip',     'LEye',     'LKnee',     'LElbow',     'Neck',
        'RElbow',     'RKnee',     'LEar',     'RShoulder',     'REar',
        'LWrist',     'LShoulder',     'LAnkle',     'REye',     'RWrist',
        'RHip',     'RAnkle',     'Nose'
        ]

    scaleFactor = 1.2

    data_root = '/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/images'

    annotations_csv = pd.read_csv('/proj/berzelius-2024-331/users/x_hensh/git/OSX/dataset/IMA/pose_estimates_youtube_dataset.csv')
    
    image_shapes_, video_names_, imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], [], [], []
    for video_name in video_names_train:
        annotations_video = annotations_csv[annotations_csv.video == video_name].reset_index(drop=True)
        frame_names = set(annotations_video.frame)
        print (video_name)
        for frame_name in frame_names:
            annotations_frame = annotations_video[annotations_video.frame==frame_name].reset_index(drop=True)
            imgname = f"1{video_name[-6:]}{frame_name:06d}.jpg"
            
            img_path = osp.join(data_root, imgname)
            if not osp.exists(img_path): continue

            body_parts = set(annotations_frame.bp) if body_parts is None else body_parts
            n_row, _ = annotations_frame.shape

            keypoint_buf = list()

            for body_part in body_parts:
                keypoint_buf.append([annotations_frame[annotations_frame.bp==body_part].reset_index().x[0],
                                    annotations_frame[annotations_frame.bp==body_part].reset_index().y[0]])

            keypoint_buf = np.array(keypoint_buf)
            keypoint_validx = keypoint_buf[:,0]
            keypoint_validy = keypoint_buf[:,1]
            ind_validx = np.isfinite(keypoint_validx)
            ind_validy = np.isfinite(keypoint_validy)

            keypoints = np.zeros([keypoint_buf.shape[0], 3])
            keypoints[:,0:2] = keypoint_buf
            keypoints[ind_validx * ind_validy,2] = 1

            keypoint_validx = keypoint_validx[ind_validx]
            keypoint_validy = keypoint_validy[ind_validy]

            if (len(keypoint_validx) * len(keypoint_validy) == 0): continue
            box = [min(keypoint_validx), min(keypoint_validy),
                   max(keypoint_validx), max(keypoint_validy)]
            
            center = [(box[2] + box[0])/2.0, (box[3] + box[1])/2.0]

            scale = scaleFactor * max(box[2]-box[0], box[3] - box[1])/200




            video_names_.append(video_name)
            imgnames_.append(imgname)
            parts_.append(np.array(keypoints))
            image_shapes_.append(video_sizes_dict[video_name])
            centers_.append(center)
            scales_.append(scale)


    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'ima_train.npz')
    np.savez(out_file, 
             video_name=video_names_,
             imgname=imgnames_,
             part=parts_,
             scale=scales_,
             center=centers_,
             image_shape=image_shapes_)
    


    image_shapes_, video_names_, imgnames_, scales_, centers_, parts_, openposes_ = [], [], [], [], [], [], []
    for video_name in video_names_test:
        annotations_video = annotations_csv[annotations_csv.video == video_name].reset_index(drop=True)
        frame_names = set(annotations_video.frame)
        print (video_name)
        for frame_name in frame_names:
            annotations_frame = annotations_video[annotations_video.frame==frame_name].reset_index(drop=True)
            imgname = f"1{video_name[-6:]}{frame_name:06d}.jpg"

            img_path = osp.join(data_root, imgname)
            if not osp.exists(img_path): continue

            body_parts = set(annotations_frame.bp) if body_parts is None else body_parts
            n_row, _ = annotations_frame.shape

            keypoint_buf = list()

            for body_part in body_parts:
                keypoint_buf.append([annotations_frame[annotations_frame.bp==body_part].reset_index().x[0],
                                    annotations_frame[annotations_frame.bp==body_part].reset_index().y[0]])


            keypoint_buf = np.array(keypoint_buf)
            keypoint_validx = keypoint_buf[:,0]
            keypoint_validy = keypoint_buf[:,1]
            ind_validx = np.isfinite(keypoint_validx)
            ind_validy = np.isfinite(keypoint_validy)

            keypoints = np.zeros([keypoint_buf.shape[0], 3])
            keypoints[:,0:2] = keypoint_buf
            keypoints[ind_validx * ind_validy,2] = 1

            keypoint_validx = keypoint_validx[ind_validx]
            keypoint_validy = keypoint_validy[ind_validy]

            if (len(keypoint_validx) * len(keypoint_validy) == 0): continue
            box = [min(keypoint_validx), min(keypoint_validy),
                   max(keypoint_validx), max(keypoint_validy)]
            
            center = [(box[2] + box[0])/2.0, (box[3] + box[1])/2.0]

            scale = scaleFactor * max(box[2]-box[0], box[3] - box[1])/200


            video_names_.append(video_name)
            imgnames_.append(imgname)
            parts_.append(np.array(keypoints))
            image_shapes_.append(video_sizes_dict[video_name])
            centers_.append(center)
            scales_.append(scale)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'ima_test.npz')
    np.savez(out_file, 
             video_name=video_names_,
             imgname=imgnames_,
             part=parts_,
             scale=scales_,
             center=centers_,
             image_shape=image_shapes_)

