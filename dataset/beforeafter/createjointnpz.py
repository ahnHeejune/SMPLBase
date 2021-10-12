'''   Check the inside of files

  (c) 11 Oct 2021 heejune@snut.ac.kr
    
  npz file is simply zip (not compressed) file of npy files 

'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cv2
import time
import sys

           
def convert_joints_to_smpl(joints, npzfilepath, W = 400.0, H = 600.0):
    
    '''
        convert np joints to npz file for SMPLify fromat
        joint[img, joint(14), x/y/conf]        range W, H         
        smpl_joint_file[x/y/conf, joint, img]  joint in range W, H
        
    '''
    # 1. load original joint npy file   
    print("input shape:", joints.shape) 
    
    if False:
        for i in range(joints.shape[0]):
            for j in range(joints.shape[1]):
                joints[i,j,0] = int(joints[i,j,0]*W + W/2.0) 
                joints[i,j,1] = int(joints[i,j,1]*H + H/2.0) 
        
    # 2. convert the index order
    newjoints = np.swapaxes(joints, 0,2) # change img and x,y index 
    print("output shape:", newjoints.shape)

    # 3 save it as npz file 
    np.savez (npzfilepath, est_joints = newjoints)  # 
    
    
if __name__ == "__main__":

    '''
    0,   # Right ankle  1,   # Right knee  2,   # Right hip
    3,   # Left hip     4,   # Left knee   5,   # Left ankle
    6,   # Right wrist  7,   # Right elbow  8,   # Right shoulder
    9,   # Left shoulder 10,  # Left elbow  11,  # Left wrist
    12,  # Neck       13,  # Head top
    '''             
            
    c_i = 1.0  # confidence in visible         
    joints = np.array( [ 
                            [   [ 157, 487, 1.0], [ 157, 380,  1.0], [167, 288, 1.0], 
                                [ 210, 288, 1.0], [ 220, 380, 1.0],  [220, 487, 1.0], 
                                [ 104, 290, 1.0], [ 126 , 240, 1.0], [150, 180, 1.0], 
                                [ 226, 180, 1.0], [ 252 , 240, 1.0], [270, 290, 1.0],
                                [ 189, 154, 1.0], [ 188 , 102, 1.0]],  # front fat
                            [   [ 168, 450, 1.0], [ 174, 367,  1.0], [182, 277, 1.0],
                                [ 219, 277, 1.0], [ 225, 367, 1.0],  [223, 450, 1.0], 
                                [ 127, 282, 1.0], [ 147 , 234, 1.0], [ 171, 174, 1.0],
                                [ 242, 174, 1.0], [ 255, 234, 1.0],  [265, 282, 1.0],
                                [ 205, 147, 1.0], [ 204 , 99, 1.0]],  # front slim 
                            [   [ 211, 488, c_i], [ 210, 393,  c_i], [ 196, 306, c_i], 
                                [ 196, 306, 1.0], [ 210, 393, 1.0],  [211, 488, 1.0], 
                                [ 202, 303, c_i], [ 203, 234, c_i], [200, 158, c_i], 
                                [ 200, 158, 1.0], [ 203, 234, 1.0], [202, 303, 1.0],
                                [ 182, 136, 1.0], [ 199, 82, 1.0]],  # side fat 
                            [   [ 210, 477, c_i], [ 201, 373, c_i], [194, 294, c_i], 
                                [ 184, 294, 1.0], [ 201, 373, 1.0],  [210, 477, 1.0], 
                                [ 187, 292, c_i], [ 188, 229, c_i], [193, 155, c_i], 
                                [ 193, 155, 1.0], [ 188 , 229, 1.0], [187, 292, 1.0],
                                [ 178, 131, 1.0], [ 196 , 74, 1.0]],  # side slim 
                        ])
                        
    convert_joints_to_smpl(joints, "bf_est_joints.npz", W = 400.0, H = 600.0)
   
        
        