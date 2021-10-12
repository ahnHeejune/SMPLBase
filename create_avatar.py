'''
 Image2Animation using SMPLify human reverse-rendering   
 --------------------------------------------------------

 (c) copyright 2019 heejune@seoultech.ac.kr

 Prerequisite: SMPL model 
 In : Input image 
         - 2d joint estimated
         - SMPL parameters (cam, shape, pose)  
         - segmenation mask of human object
         - cloth vertices from body 
      BVH file 

      background image file 
 Out:  
      animating video file 

'''
from __future__ import print_function 
import sys
from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
#import cPickle as pickle   # Python 2
import _pickle as pickle   # Python 3
import time
import cv2
import numpy as np
import chumpy as ch
from opendr.camera import ProjectPoints

from smpl.serialization import load_model
from smpl.verts import verts_core
from smpl.verts import verts_decorated

#from render_model import render_model

import inspect  # for debugging
import matplotlib.pyplot as plt

#from opendr.lighting import SphericalHarmonics
#from opendr.geometry import VertNormals, Rodrigues


from bvh import Bvh, BvhNode
import math

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from smpl2cloth import smpl2cloth3dcore

# this texture renderer doesnot have lightening effects
#from smpl2cloth import build_texture_renderer 

_clothlabeldict = {"background": 0,
            "hat" :1,
            "hair":2,
            "sunglass":3, #       3
            "upper-clothes":4,  #  4
            "skirt":5 ,  #          5
            "pants":6 ,  #          6
            "dress":7 , #          7
            "belt": 8 , #           8
            "left-shoe": 9, #      9
            "right-shoe": 10, #     10
            "face": 11,  #           11
            "left-leg": 12, #       12
            "right-leg": 13, #      13
            "left-arm": 14,#       14
            "right-arm": 15, #      15
            "bag": 16, #            16
            "scarf": 17 #          17
        }


# 
# print pose with annotation 
#  pose: np for all 23  
#
#               - 13 - 16 [lsh] - 18 [lelb] - 20 [lwr] - 22
#     3 - 6 - 9 - 12 - 15
#               - 14 - 17 [rsh] - 19 [relb] - 21 [rwr] - 23
# 0 -
#     1 [lhip] - 4 [lknee] - 7 [lankle] - 10
#     2 [rhip] - 5 [rknee] - 8 [rankle] - 11
#
jointname = {  0: 'root',
                1: 'lhip', 
                2: 'rhip', 
                4: 'lknee',
                5: 'rknee',
                7: 'lankle',
                8: 'rankle',
                10: 'lfoot',
                11: 'rfoot',
                3: 'lowback',
                6: 'midback',
                9: 'chest',
                12: 'neck',
                15: 'head',
                13: 'lcollar',
                14: 'rcollar',
                16: 'lsh',
                17: 'rsh',
                18: 'lelbow',
                19: 'relbow',
                20: 'lwrist',
                21: 'rwrist',
                22: 'lhand',
                23: 'rhand'}


#######################################################################################
# load dataset dependent files and call the core processing 
#---------------------------------------------------------------
# smpl_mdoel: SMPL 
# inmodel_path : smpl param pkl file (by SMPLify) 
# inimg_path: input image 
# mask image 
# joint 2d
# ind : image index 
#######################################################################################

#
# read smpl param, mask, original images



###############################################################################
# restore the Template posed vertices  
# 
# return: vertices (ccoordinates), jtrs' locations
# 
# multistep joint reverse method   
#    
#                           - 13        - 16 [lsh] - 18 [lelb] - 20 [lwr] - 22
#     3        - 6          - 9         - 12       - 15
#                           - 14        - 17 [rsh] - 19 [relb] - 21 [rwr] - 23
# 0 - 
#     1 [lhip] - 4 [lknee] - 7 [lankle] - 10
#     2 [rhip] - 5 [rknee] - 8 [rankle] - 11                
###############################################################################
def restorePose(smpl_model, vertices, j_transformed, pose_s):

    joint_hierarchy = [ [0],    # the level not exactly hierarchy 
                        [1, 2, 3], 
                        [6, 4, 5], 
                        [7, 8, 9, 13, 14], 
                        [10, 11, 12, 16, 17], 
                        [15, 18, 19], 
                        [20, 21], 
                        [22, 23] ] 

    # constant terms 
    weights = ch.array(smpl_model.weights.r)
    kintree_table = smpl_model.kintree_table.copy()

    # 2. back to the default pose
    ##########################################################################
    for step in range(0, len(joint_hierarchy)):

        # 1. get the vertices and paramters 
        if step == 0:
            v_posed = ch.array(vertices)
            J = ch.array(j_transformed)
        else:
            v_posed = ch.array(v_cur.r)
            J = ch.array(jtr.r)

        pose =  ch.zeros(smpl_model.pose.size)

        # 2. LBS setup 
        v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J,
                            weights = weights,
                            kintree_table = kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')
        # 3. Renderer 
        #cam_s.v = v_cur

        # 4. repose 
        for joint in joint_hierarchy[step]:
            pose[joint*3:(joint+1)*3] = - pose_s[joint*3:(joint+1)*3]     

    return  v_cur.r, jtr.r 


################################################################################
# building LBS
#
################################################################################
def buildLBS(smpl_model, vertices, jtr):

    # constant terms 
    weights = ch.array(smpl_model.weights.r)
    kintree_table = smpl_model.kintree_table.copy()

    # 1. get the vertices and paramters 
    v_posed = ch.array(vertices)
    J = ch.array(jtr)
    pose =  ch.zeros(smpl_model.pose.size)

    # 2. LBS setup 
    v_cur, jtr = verts_core(  pose = pose,
                            v    = v_posed,
                            J    = J,
                            weights = weights,
                            kintree_table = kintree_table,
                            xp = ch, #ch vs np
                            want_Jtr = True,
                            bs_style = 'lbs')

    return  v_cur, jtr, pose 


def recons_human(   smpl_model, 
                    params, mask, img, j2d,    
                    img_bg = None):
                   
    idx = 0               
    # @TODO Processing the mask easy to matching from SMPL
    #  pre-processing for boundary matching  
    mask[mask == _clothlabeldict['bag']] = 0  # remove bag
    # cut the connected legs 
    if idx == 0:
        mask[500:,190] = 0
    elif idx == 1:
        mask[500:,220] = 0
    else:
        print('Not prepared Yet for the idx!')
        #, exit()
  
    # 1. SMPL body to cloth  
    pose  = params['pose']          # angles ((23+1)x3), numpy
    betas = params['betas']
    n_betas = betas.shape[0]        # 10, tuple to numpy
    
    cam_f = params['cam_f']
    cam_rt = params['cam_rt']
    cam_t = params['cam_t']    
    cam_c = params['cam_c']
    cam_k = params['cam_k']
    cam   =  ProjectPoints(f=cam_f, rt=cam_rt, t=cam_t, k=cam_k, c=cam_c)
     
    joint_transformed, vt, ft, texture = smpl2cloth3dcore(cam,      # camera model, Ch
                 betas,    # shape coeff, numpy
                 n_betas,  # num of PCA
                 pose,     # angles, 27x3 numpy
                 img,    # img numpy
                 mask,     # mask 
                 j2d, 
                 smpl_model,
                 bHead = True,
                 viz =False)

    original_projection = cam.r.copy()

    # 2. restore default posed vertices 
    vertices, jtr = restorePose(smpl_model, cam.v.r, joint_transformed, pose)  
   
    # 3 build model
    v_cur, jtr_cur, pose = buildLBS(smpl_model, vertices, jtr)

    '''
    # 4 build texture renderer 
    h, w = mask_s.shape[:2]
    cam_s.v = v_cur 
    texture_renderer = build_texture_renderer(cam_s, v_cur, 
            smpl_model.f, vt, ft, texture[::-1, :,:], w, h, 1.0, near=0.5, far = 125., background_image = img_bg[:,:,::-1])  
    
    return cam_s, pose, texture_renderer
    '''
    
    return cam, v_cur, original_projection

    
    '''
    # 5 re-pose
    cam_s.v = v_cur  
    pose[:] = pose_t[:]

    # 6 visualize 
    img = (texture_renderer.r*255.0).astype('uint8')
    # joint display 
    cam_s.v = jtr_cur.r  
    j2d = cam_s.r.copy()
    for i in range(j2d.shape[0]):
        cv2.drawMarker(img, (int(j2d[i,0]), int(j2d[i,1])), (255,0,0), cv2.MARKER_DIAMOND, 5, 3)
    plt.subplot(1,2,2)
    plt.imshow(img)
    plt.axis('off')
    plt.draw()
    plt.show()
    _ = raw_input()
    '''

    
def load_SMPL_model(MODEL_DIR, sex = 2): # 0 neutral, 1: male,  2: female 
  
    if sex == 0:    
        MODEL_NEUTRAL_PATH = join(MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
        model = load_model(MODEL_NEUTRAL_PATH)    
    elif sex == 1:
        MODEL_MALE_PATH = join(MODEL_DIR, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')
        model = load_model(MODEL_MALE_PATH)
    elif sex == 2:
        MODEL_FEMALE_PATH = join(MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
        model = load_model(MODEL_FEMALE_PATH)
    else:    
        print("No sex given")
        model = None
    return model    


                   
Limbs = [ (13,12),  # head to check/neck
          (12, 8), (12, 9), # chick to shoulders
          (8,  7), (9, 10), # shoulders to elbows
          (7,  6), (10,11), # elbows to wrists
          (8,  2), (9,  3), # shoulders to hips
          (2,  1), (3,  4), # hips to knees
          (1,  0), (4,  5)] # knees to ankles 
 
                   
# overlay joints 
def drawJoints(img, joints, jointInPixel = False):
    
    if len(img.shape) == 2:  # gray 
        color = 0        # black 
    else:                    # rgb
        color = (0,0,0)  # black
    
    if jointInPixel != True:
        height_scale = img.shape[0]
        width_scale = img.shape[1]
        #print('h:', height, 'w:', width)
    else:
        height_scale = 1.0
        width_scale  = 1.0
        
    for i in range(len(joints)):
        x = int((joints[i,0] +0.499)*width_scale)
        y = int((joints[i,1] +0.499)*height_scale)
        cv2.circle(img, (x,y), 4, color)
        
# overlay limbs 
def drawLimbs(img, joints, jointInPixel = False):
   
    if len(img.shape) == 2:  # gray 
        color = 0        # black 
    else:                    # rgb
        color = (0,0,0)  # black
        
    if jointInPixel != True:
        height_scale = img.shape[0]
        width_scale = img.shape[1]
        #print('h:', height, 'w:', width)
    else:
        height_scale = 1.0
        width_scale  = 1.0
        
    for limb in Limbs:
        x = int((joints[limb[0],0] +0.5)*width_scale)
        y = int((joints[limb[0],1] +0.5)*height_scale)
        pt1 = (x,y)
        x = int((joints[limb[1],0] +0.5)*width_scale)
        y = int((joints[limb[1],1] +0.5)*height_scale)
        pt2 = (x,y)
        
        cv2.line(img, pt1, pt2, color, 2) # 0 : black
        

if __name__ == '__main__':

    if len(sys.argv) < 4:
       print('current usage: %s . 10k 0'% sys.argv[0])
       print('original-usage: %s base_dir dataset idx'% sys.argv[0]), exit()

    # 1. directory check and setting
    base_dir = abspath(sys.argv[1])
    #print(base_dir)
    dataset = sys.argv[2]
    idx = int(sys.argv[3])
    
    if not exists(base_dir):
        print('No such a directory for base', base_path, base_dir), exit()

    # 1.1 Loading SMPL models (independent from dataset)
    MODEL_DIR = join(base_dir, 'models')
    smpl_model = load_SMPL_model(MODEL_DIR, 2)
     
    data_dir =  join(base_dir, 'dataset', dataset)
    
    # 1.2 model params 
    smpl_param_path = join(data_dir, 'smplified' , '%04d.pkl'%idx) 
    with open(smpl_param_path, 'rb') as f:
        if f is None:
            print("cannot open", smpl_param_path), exit()
        params = pickle.load(f)
    #_examine_smpl_params(params)

    # 1.3 2d rgb image for texture
    human_img_path = join(data_dir, 'input', 'dataset%s_%04d.jpg'%(dataset,idx)) 
    human_img = cv2.imread(human_img_path)
    if human_img is None:
        print("cannot open", human_img_path), exit()

    # 1.4 segmentation mask 
    human_mask_path = join(data_dir, 'parsed', 'dataset%s_%04d.png'%(dataset,idx)) 
    human_mask = cv2.imread(human_mask_path, cv2.IMREAD_UNCHANGED)
    if human_mask is None:
        print("cannot open",  human_mask_path), exit()
    
    # 1.5 joint needed for matching process, ^^;;
    estj2d = np.load(join(data_dir, 'est_joints.npz'))['est_joints']
    j2d = estj2d[:2, :, idx].T
   
   
    # 2. image to 3d model
    cam, v_cur, projected_v_old = recons_human(smpl_model, params, human_mask, human_img, j2d, img_bg = None)


    # 3. projection and visualization 
    w = human_mask.shape[1]
    h = human_mask.shape[0]
   
    # before reposed 
    print(type(projected_v_old), projected_v_old.shape)
    img_projected_old = np.zeros((h,w,3))  # float ?
    # mapping points into the 
    for i in range(projected_v_old.shape[0]):
        x, y = int(projected_v_old[i, 0]), int(projected_v_old[i,1])
        if x >= 0 and x < w and y >= 0 and y < h:
            img_projected_old[y, x, :] = 1.0 


    # after reposed
    cam.v = v_cur.r
    print(type(cam.r), cam.r.shape)
    img_projected = np.zeros((h,w,3))  # float ?
    # mapping points into the 
    for i in range(cam.r.shape[0]):
        x, y = int(cam.r[i, 0]), int(cam.r[i,1])
        if x >= 0 and x < w and y >= 0 and y < h:
            img_projected[y, x, :] = 1.0 
  
  
    drawJoints(human_img, j2d, True)
    drawLimbs(human_img, j2d, True)
    plt.subplot(1,4,1),plt.imshow(human_img[:,:,::-1]), plt.axis('off'), plt.title('input')
    plt.subplot(1,4,2),plt.imshow(human_mask != 0), plt.axis('off'), plt.title('mask') 
    plt.subplot(1,4,3),plt.imshow(img_projected_old[:,:,:]), plt.axis('off'), plt.title('before reposed')  # why not up side down??
    plt.subplot(1,4,4),plt.imshow(img_projected[::-1,:,:]), plt.axis('off'), plt.title('after reposed')  # y-axis upside-down
    plt.show() 
 
 