import os
import sys
import cv2
import glob
#import h5py
import json
import numpy as np
import scipy.io as sio
import scipy.misc
#from .read_openpose import read_openpose
import dsr.core.config as cfg

def read_calibration(calib_file, vid_list):
    Ks, Rs, Ts = [], [], []
    file = open(calib_file, 'r')
    content = file.readlines()
    for vid_i in vid_list:
        K = np.array([float(s) for s in content[vid_i*7+5][11:-2].split()])
        K = np.reshape(K, (4, 4))
        RT = np.array([float(s) for s in content[vid_i*7+6][11:-2].split()])
        RT = np.reshape(RT, (4, 4))
        R = RT[:3,:3]
        T = RT[:3,3]/1000
        Ks.append(K)
        Rs.append(R)
        Ts.append(T)
    return Ks, Rs, Ts
    
def train_data(dataset_path, openpose_path, out_path, joints_idx, scaleFactor, extract_img=False, fits_3d=None):

    joints17_idx = [4, 18, 19, 20, 23, 24, 25, 3, 5, 6, 7, 9, 10, 11, 14, 15, 16]

    h, w = 2048, 2048
    imgnames_, scales_, centers_ = [], [], []
    parts_, Ss_, openposes_ = [], [], []

    # training data
    user_list = range(1,9)
    seq_list = range(1,3)
    vid_list = list(range(3)) + list(range(4,9))

    counter = 0

    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            # mat file with annotations
            annot_file = os.path.join(seq_path, 'annot.mat')
            annot2 = sio.loadmat(annot_file)['annot2']
            annot3 = sio.loadmat(annot_file)['annot3']
            # calibration file and camera parameters
            calib_file = os.path.join(seq_path, 'camera.calibration')
            Ks, Rs, Ts = read_calibration(calib_file, vid_list)

            for j, vid_i in enumerate(vid_list):

                # image folder
                imgs_path = os.path.join(seq_path,    
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                if extract_img:

                    # if doesn't exist
                    if not os.path.isdir(imgs_path):
                        os.makedirs(imgs_path)

                    # video file
                    vid_file = os.path.join(seq_path,
                                            'imageSequence',
                                            'video_' + str(vid_i) + '.avi')
                    vidcap = cv2.VideoCapture(vid_file)

                    # process video
                    frame = 0
                    while 1:
                        # extract all frames
                        success, image = vidcap.read()
                        if not success:
                            break
                        frame += 1
                        # image name
                        imgname = os.path.join(imgs_path,
                            'frame_%06d.jpg' % frame)
                        # save image
                        cv2.imwrite(imgname, image)

                # per frame
                cam_aa = cv2.Rodrigues(Rs[j])[0].T[0]
                pattern = os.path.join(imgs_path, '*.jpg')
                img_list = glob.glob(pattern)
                for i, img_i in enumerate(img_list):

                    # for each image we store the relevant annotations
                    img_name = img_i.split('/')[-1]
                    img_view = os.path.join('S' + str(user_i),
                                            'Seq' + str(seq_i),
                                            'imageFrames',
                                            'video_' + str(vid_i),
                                            img_name)
                    joints = np.reshape(annot2[vid_i][0][i], (28, 2))[joints17_idx]
                    S17 = np.reshape(annot3[vid_i][0][i], (28, 3))/1000
                    S17 = S17[joints17_idx] - S17[4] # 4 is the root
                    bbox = [min(joints[:,0]), min(joints[:,1]),
                            max(joints[:,0]), max(joints[:,1])]
                    center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
                    scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

                    # check that all joints are visible
                    x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
                    y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
                    ok_pts = np.logical_and(x_in, y_in)
                    if np.sum(ok_pts) < len(joints_idx):
                        continue
                        
                    part = np.zeros([24,3])
                    part[joints_idx] = np.hstack([joints, np.ones([17,1])])
                    json_file = os.path.join(openpose_path, 'mpi_inf_3dhp',
                        img_view.replace('.jpg', '_keypoints.json'))
                    openpose = read_openpose(json_file, part, 'mpi_inf_3dhp')

                    S = np.zeros([24,4])
                    S[joints_idx] = np.hstack([S17, np.ones([17,1])])

                    # because of the dataset size, we only keep every 10th frame
                    counter += 1
                    if counter % 10 != 1:
                        continue

                    # store the data
                    imgnames_.append(img_view)
                    centers_.append(center)
                    scales_.append(scale)
                    parts_.append(part)
                    Ss_.append(S)
                    openposes_.append(openpose)
                       
    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_inf_3dhp_train.npz')
    if fits_3d is not None:
        fits_3d = np.load(fits_3d)
        np.savez(out_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           part=parts_,
                           pose=fits_3d['pose'],
                           shape=fits_3d['shape'],
                           has_smpl=fits_3d['has_smpl'],
                           S=Ss_,
                           openpose=openposes_)
    else:
        np.savez(out_file, imgname=imgnames_,
                           center=centers_,
                           scale=scales_,
                           part=parts_,
                           S=Ss_,
                           openpose=openposes_)        
        
        
def test_data(dataset_path, out_path, joints_idx, scaleFactor):

    joints17_idx = [14, 11, 12, 13, 8, 9, 10, 15, 1, 16, 0, 5, 6, 7, 2, 3, 4]

    imgnames_, scales_, centers_, parts_,  Ss_ = [], [], [], [], []

    # training data
    user_list = range(1,7)

    for user_i in user_list:
        seq_path = os.path.join(dataset_path,
                                'mpi_inf_3dhp_test_set',
                                'TS' + str(user_i))
        # mat file with annotations
        annot_file = os.path.join(seq_path, 'annot_data.mat')
        mat_as_h5 = h5py.File(annot_file, 'r')
        annot2 = np.array(mat_as_h5['annot2'])
        annot3 = np.array(mat_as_h5['univ_annot3'])
        valid = np.array(mat_as_h5['valid_frame'])
        for frame_i, valid_i in enumerate(valid):
            if valid_i == 0:
                continue
            img_name = os.path.join('mpi_inf_3dhp_test_set',
                                   'TS' + str(user_i),
                                   'imageSequence',
                                   'img_' + str(frame_i+1).zfill(6) + '.jpg')

            joints = annot2[frame_i,0,joints17_idx,:]
            S17 = annot3[frame_i,0,joints17_idx,:]/1000
            S17 = S17 - S17[0]

            bbox = [min(joints[:,0]), min(joints[:,1]),
                    max(joints[:,0]), max(joints[:,1])]
            center = [(bbox[2]+bbox[0])/2, (bbox[3]+bbox[1])/2]
            scale = scaleFactor*max(bbox[2]-bbox[0], bbox[3]-bbox[1])/200

            # check that all joints are visible
            img_file = os.path.join(dataset_path, img_name)
            I = scipy.misc.imread(img_file)
            h, w, _ = I.shape
            x_in = np.logical_and(joints[:, 0] < w, joints[:, 0] >= 0)
            y_in = np.logical_and(joints[:, 1] < h, joints[:, 1] >= 0)
            ok_pts = np.logical_and(x_in, y_in)
            if np.sum(ok_pts) < len(joints_idx):
                continue

            part = np.zeros([24,3])
            part[joints_idx] = np.hstack([joints, np.ones([17,1])])

            S = np.zeros([24,4])
            S[joints_idx] = np.hstack([S17, np.ones([17,1])])

            # store the data
            imgnames_.append(img_name)
            centers_.append(center)
            scales_.append(scale)
            parts_.append(part)
            Ss_.append(S)

    # store the data struct
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    out_file = os.path.join(out_path, 'mpi_inf_3dhp_test.npz')
    np.savez(out_file, imgname=imgnames_,
                       center=centers_,
                       scale=scales_,
                       part=parts_,
                       S=Ss_)    

import os
def mini_rgbd_web_extract(out_path, subjects):


    joint_list = [
        'global_',
        'leftThigh',
        'rightThigh',
        'spine',
        'leftCalf',
        'eightCalf',
        'spine1',
        'leftFoot',
        'rightFoot',
        'spine2',
        'leftToes',
        'rightToes',
        'neck',
        'leftShoulder',
        'rightShoulder',
        'head',
        'leftUpperArm',
        'rightUpperArm',
        'leftForeArm',
        'rightForeArm',
        'leftHand',
        'rightHand',
        'leftFingers',
        'rightFingers',
        'noseVertex'
    ]

    mini_to_smil_map = {
        'global_': 'pelvs',                 # 0
        'leftThigh': 'left_hip',            # 1
        'rightThigh': 'right_hip',          # 2
        'spine': 'spine1',                  # 3
        'leftCalf'  :'left_knee',           # 4
        'eightCalf' :'right_knee',          # 5
        'spine1':'spine2',                  # 6
        'leftFoot':'left_ankle',            # 7
        'rightFoot':'right_ankle',          # 8
        'spine2':'spine3',                  # 9
        'leftToes':'left_foot',             # 10
        'rightToes':'right_foot',           # 11
        'neck':'neck',                      # 12
        'leftShoulder':'left_collar',       # 13
        'rightShoulder':'right_collar',     # 14
        'head':'head',                      # 15
        'leftUpperArm':'left_shoulder',     # 16
        'rightUpperArm':'right_shoulder',   # 17
        'leftForeArm':'left_elbow',         # 18
        'rightForeArm':'right_elbow',       # 19
        'leftHand':'left_wrist',            # 20
        'rightHand':'right_wrist',          # 21
        'leftFingers':'left_hand',          # 22
        'rightFingers':'right_hand',        # 23
        'noseVertex':'nose'                 # 24
    }

    scaleFactor = 1.2

    imagenames_, scales_, centers_, parts_, S_, openposes_ = [], [], [], [], [], []
    pose_, trans, shape_ = [], [], []
    has_smpls_ = []
    subjects_ = []

    trans_ = []
    user_list = subjects
    data_root = 'D:\work\data\MINI-RGBD_web'
    for subject_id in user_list:


        # rgb               syn_xxxxx.png
        # fg_mask           mask_xxxxx.png
        # smil_params       xxxxx_trans.txt // xxxxx_pose.txt
        # joints_2Ddep      syn_joints_2Ddep_xxxxx.txt
        # joints_3d         syn_joint_3D_xxxxx.txt
        # depth             syn_xxxxx_depth.png

        subject_image_path = os.path.join(data_root, subject_id, 'rgb')
        rgb_image_names = os.listdir(subject_image_path)
        frame_ids = [item[4:-4] for item in rgb_image_names]

        shape_file_path = os.path.join(data_root, subject_id, 'smil_shape_betas.txt')
        with open(shape_file_path, 'r') as f:
            shape = f.read().strip()
            shape = shape.split('\n')
            shape = np.array(shape, dtype='float')



        for frame_i, frame_id in enumerate(frame_ids):
            imagenames_.append(f'syn_{frame_id}.png')
            shape_.append(shape)

            joints_3D_path = os.path.join(data_root, subject_id, 'joints_3D', f'syn_joints_3D_{frame_id}.txt')
            with open(joints_3D_path, 'r') as f:
                joints_3D = f.read().strip()
                joints_3D = [item.split() for item in joints_3D.split('\n')]
                joints_3D = np.array(joints_3D, dtype='float')
            joints_3D = joints_3D[:, :-1]
            joints_3D = np.concatenate((joints_3D, np.ones([joints_3D.shape[0], 1])), axis=1)
            S_.append(joints_3D)

            joints_2Ddep_path = os.path.join(data_root, subject_id, 'joints_2Ddep', f'syn_joints_2Ddep_{frame_id}.txt')
            with open(joints_2Ddep_path, 'r') as f:
                joints_2Ddep = f.read().strip()
                joints_2Ddep = [item.split() for item in joints_2Ddep.split('\n')]
                joints_2Ddep = np.array(joints_2Ddep, dtype='float')
            joints_2Ddep = joints_2Ddep[:,:2]
            joints_2Ddep = np.concatenate((joints_2Ddep, np.ones([joints_2Ddep.shape[0], 1])), axis=1)
            parts_.append(joints_2Ddep.copy())

            pose_file_path = os.path.join(data_root, subject_id, 'smil_params', f'{frame_id}_pose.txt')
            with open(pose_file_path, 'r') as f:
                pose = f.read().strip()
                pose = pose.split('\n')
                pose = np.array(pose, dtype='float')
            pose_.append(pose)

            trans_file_path = os.path.join(data_root, subject_id, 'smil_params', f'{frame_id}_trans.txt')
            with open(trans_file_path, 'r') as f:
                trans = f.read().strip()
                trans = trans.split('\n')
                trans = np.array(trans, dtype='float')
            trans_.append(trans)


            head_margin = (joints_2Ddep[24, :-1] - joints_2Ddep[12, :-1]) * 2
            joints_2Ddep[24, :-1] = joints_2Ddep[24, :-1] + head_margin


            bbox = [min(joints_2Ddep[:, 0]), min(joints_2Ddep[:,1]),
                    max(joints_2Ddep[:, 0]), max(joints_2Ddep[:,1])]



            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            scale = scaleFactor * max(bbox[2] - bbox[0], bbox[3] - bbox[1]) / 200

            centers_.append(center)
            scales_.append(scale)
            has_smpls_.append(1)
            subjects_.append(subject_id)

    np.savez(out_path,
             imgname=imagenames_,
             center=centers_,
             scale=scales_,
             part=parts_,
             S=S_,
             pose=pose_,
             shape=shape_,
             trans_=trans_,
             has_smpl=has_smpls_,
             subjects_=subjects_, )






if __name__ == '__main__':
    # = cfg.DATASET_NPZ_PATH
    out_path = 'dsr_data/dataset_extras/minirgbd_train.npz'
    #openpose_path = cfg.OPENPOSE_PATH
    mini_rgbd_web_extract(out_path, subjects=['01','02','04','05','06','08','09','10', '12'])

    out_path = 'dsr_data/dataset_extras/minirgbd_valid.npz'
    # openpose_path = cfg.OPENPOSE_PATH
    mini_rgbd_web_extract(out_path, subjects=['03', '07', '11'])

