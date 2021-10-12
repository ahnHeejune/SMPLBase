'''
  Simple demonstration for how to use Texture

 (c) copyright 2019 heejune@seoultech.ac.kr

  SMPLified SMPL model => human image for UV-Texture 
                   projection

 1) reconstruct SMPL body with SMPLified pose, shape and camera params
 2) project onto the human images for uv-coorindates 
 3) save it the model file into Wavefront obj format 
 4) load the obj file using trimesh 
 5) rendering the model with texture with pyrender 
 
 
 The Wavefront OBJ file include 
 1) 3d vertices coorindate ('v')
 2) faces  ('f')
 3) uv-coorindate ('vt')
 (we are not using vertex normal ('vn')
 4) link to the texture (called as material)
 
 for example. smpl.obj ==> smpl.obj.mtl => data10k_0000.jpg 
                

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

import math

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import obj_viewer 

def recons_body(   smpl, 
                    params, mask, img, j2d,    
                    img_bg = None):
                  
  
    # 1. SMPL body to cloth  
    pose  = params['pose']          # angles ((23+1)x3), numpy
    betas = params['betas']
    n_betas = betas.shape[0]        # 10, tuple to numpy
    
    
    # smpl body reconstruction 
    smpl.betas[:] = betas
    smpl.pose[:] = pose
    

    cam_f = params['cam_f']
    cam_rt = params['cam_rt']
    cam_t = params['cam_t']    
    cam_c = params['cam_c']
    cam_k = params['cam_k']
    cam   =  ProjectPoints(f=cam_f, rt=cam_rt, t=cam_t, k=cam_k, c=cam_c)
   
    return cam, smpl

    
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
 
  
        
def saveMeshObjFile(objFilePath, mtlfilePath, uvfilePath, vertices, vts, faces):

    
    #saveObjfile(filename, v_cur.r, smpl_model.f, projected_v_old)
    with open( objFilePath, 'w') as fp:
        fp.write('mtllib {}\n'.format(mtlfilePath))  
        for v in vertices:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )
        for vt in vts:
            fp.write( 'vt %f %f\n' % ( vt[0], vt[1] ) )
        fp.write('usemtl {}\n'.format("material_0"))    
        for f in faces+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d/%d %d/%d %d/%d\n' %  (f[0], f[0], f[1], f[1], f[2], f[2]) ) # polygons for 3D and textture

    with open( mtlfilePath, 'w') as fp:
        fp.write('newmtl material_0\n')  
        fp.write('# shader_type beckmann\n')  
        fp.write('map_Kd {}\n'.format(uvfilePath))          
        

import trimesh
import sys 
from pyrender import  Mesh, Node, Scene, Viewer, PerspectiveCamera
def showObjFile(objFilePath, cam0):

    """  load the obj file and  render it by Pyrender 
        @TODO, use the input  camera paramters 
    
    """
    
    
    trimesh1 = trimesh.load(objFilePath)
    mesh = Mesh.from_trimesh(trimesh1)

    # Scene creation
    scene = Scene(ambient_light=np.array([1., 1.0, 1.0, 1.0])) ##### I increased this 

    scene.add(mesh)

    cam_pose = np.array([
        [1.0,  0.0,  0.0,  0.2],   # zero-rotation  
        [0.0,  1.0,  0.0,  0.0],   # 
        [0.0,  0.0,  1.0,  2.0],   # (0, 0, 2.0) translation  
        [0.0,  0.0,  0.0,  1.0]
    ])
    cam   = PerspectiveCamera(yfov=(np.pi* 75.0/180.0))     # 75 degree FOV
    cam_node = scene.add(cam, pose = cam_pose)
    
    v = Viewer(scene, shadows=False)
        
        
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
    cam, smpl = recons_body(smpl_model, params, human_mask, human_img, j2d, img_bg = None)

    # 3. projection and visualization 
    w = human_mask.shape[1]
    h = human_mask.shape[0]
   
    # UV coordinate
    cam.v = smpl.r      # projection 
    vts = cam.r.copy()
    vts[:,0] = vts[:,0]/w  # x normalization 
    vts[:,1] = 1.0 - vts[:,1]/h  # y normalization  (and upside-down)
 
    objFilePath = "smpl.obj"
    mtlfilePath = objFilePath + ".mtl"
    uvfilePath = human_img_path
    
    saveMeshObjFile(objFilePath, mtlfilePath, uvfilePath, smpl.r, vts, smpl.f)
  
    # 4. visualize
    showObjFile(objFilePath, cam)