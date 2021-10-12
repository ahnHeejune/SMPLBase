###################################################################
# TO STUDY 
###################################################################
# 3D Triangle Mesh model 
# 3D mesh model files
# SMPL model 
# OpenGL 
# PyRender 
# BVH files 
# (c) 2020  heejune@seoultech.ac.kr
###################################################################


from smpl.serialization import load_model  
import numpy as np
#from bvh import Bvh, BvhNode
import math
import trimesh
import pyrender
import time

import cv2
    
def setupScene(cam_pose):
    
    """ Setup a scene, with a fixed camera pose  """ 

    # setup scene and camera  
    scene = pyrender.Scene( bg_color=np.array([0.0, 0.0, 0.0]),  
                            ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
    cam   = pyrender.PerspectiveCamera(yfov=(np.pi* 75.0/ 180.0))     # 90 degree FOV
    cam_node = scene.add(cam, pose = cam_pose)
        
    viewer = pyrender.Viewer(scene, use_raymond_lighting=True,  run_in_thread=True)  # a separate thread for rendering
    
    return scene, viewer
    
    
    

# global variables

smplMeshNode = None
materialTexture = None

def updateSMPLwTexture(scene, viewer, vertices, faces):

    global materialTexture, smplMeshNode

    if materialTexture is None:
        tex =  cv2.cvtColor(cv2.imread('texture.jpg'), cv2.COLOR_BGR2RGB)
        tex = pyrender.Texture(source=tex, source_channels='RGB')
        materialTexture = pyrender.material.MetallicRoughnessMaterial(baseColorTexture=tex, wireframe=False)
 
    triMesh = trimesh.Trimesh(vertices, faces)
    mesh = pyrender.Mesh.from_trimesh(triMesh, material = materialTexture)   
    
    # update mesh 
    viewer.render_lock.acquire()
    if smplMeshNode is not None:
        scene.remove_node(smplMeshNode)
    smplMeshNode = scene.add(mesh)
    viewer.render_lock.release()
    

faceColors  = None

def updateSMPLFace(scene, viewer, vertices, faces):

    global faceColors, smplMeshNode

    
    """ update the smpl mesh
        Currently, remove the existing one if any and add new one 
        TODO: can we just upate the vertices info keeping the mesh object is same?
        https://pyrender.readthedocs.io/en/latest/examples/viewer.html
    """ 
    # face-color used
    if faceColors is None:
        faceColors = np.ones([faces.shape[0], 4]) * [0.9, 0.5, 0.5, 1.0]     # pinky skin               
    
    triMesh = trimesh.Trimesh(vertices, faces, face_colors=faceColors)
        

    '''
    texture = None, f_colors = None, v_colors = None):     
        # 1. color setting     
        if texture is not None:
            raise Exception('not yet support texture (TODO)')
        elif f_colors is not None:
            triMesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors)
        elif v_colors is not None:
            v_colors = np.ones([vertices.shape[0], 4]) * [0.7, 0.7, 0.7, 0.8]   
            triMesh = trimesh.Trimesh(vertices, faces, vertex_colors=v_colors)
        else:
            f_colors = np.ones([faces.shape[0], 4]) * [0.9, 0.5, 0.5, 1.0]                  
            triMesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors)
    '''        
    
    # build a new mesh 
    mesh = pyrender.Mesh.from_trimesh(triMesh, smooth = False)   
    
    # update mesh 
    viewer.render_lock.acquire()
    if smplMeshNode is not None:
        scene.remove_node(smplMeshNode)
    smplMeshNode = scene.add(mesh)
    viewer.render_lock.release()
    
    """
    plot_joints = False  # TODO
    if plot_joints:
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)
    """
  

def updateSMPL(scene, viewer, vertices, faces):

    updateSMPLFace(scene, viewer, vertices, faces)
    
    #updateSMPLwTexture(scene, viewer, vertices, faces)
    


#
# update SMPL pose   
#
def poseSMPL(cam_pose, smpl, pose, viz = False):

    smpl.pose[:] = pose.flatten()
    
    if viz:
        visualize(cam_pose, smpl.r, smpl.f) 

    
'''
def runBVH(smpl, motion_file_path):

    
    with open(motion_file_path) as f:
            mocap = Bvh(f.read())
    
    joints = mocap.get_joints_names()
    print('# of joints:', len(joints), joints)
    print('# of frames:', mocap.nframes)
    allchannel = [0,1,2]  
    
    pose = np.zeros([24,3])

    #
    #           - 13 - 16 [lsh] - 18 [lelb] - 20 [lwr] - 22
    #       3 - 6 - 9 - 12 - 15
    #           - 14 - 17 [rsh] - 19 [relb] - 21 [rwr] - 23
    # 0 - 
    #       1 [lhip] - 4 [lknee] - 7 [lankle] - 10
    #       2 [rhip] - 5 [rknee] - 8 [rankle] - 11  
    #

    bvh2smpl_jtmap = {  'Hips': 0,
                    'Chest':6,
                    'Chest2':9,
                    'LeftCollar':13,
                    'LeftShoulder': 16,
                    'LeftElbow': 18,
                    'LeftWrist': 20,
                    'RightCollar':14,
                    'RightShoulder':17,
                    'RightElbow':19,
                    'RightWrist':21,
                    'Neck':12,
                    'Head':15, # not correct but..
                    'LeftHip':1,
                    'LeftKnee':4,
                    'LeftAnkle':7,
                    'RightHip':2,
                    'RightKnee':5, 
                    'RightAnkle':8 }

  
    cam_pose = np.array([
        [1.0,  0.0,  0.0,  0.0],   # 90-rotation around z axis 
        [0.0,  1.0,  0.0,    0.0],   # 
        [0.0,  0.0,  1.0,   2.0],   # (0, 0, 0.0) translation  
        [0.0,  0.0,  0.0,   1.0]
    ])
  
    for  fn in range(mocap.nframes):
        print('fn:', fn);
        pose[:,:] = 0.0 # reset 
        for joint in joints:
            angs = mocap.frame_joint_channels(fn, joint, allchannel)
            #print('\t', joint, ':', math.radians(angs[0]), math.radians(angs[1]), math.radians(angs[2]) )
            smpljt = bvh2smpl_jtmap[joint]
            pose[smpljt,:] = (math.radians(angs[0]),math.radians(angs[1]), math.radians(angs[2]) )
            poseSMPL(cam_pose, smpl, pose, True)
'''   


def save_to_objfile(m):

    ## 3. Write the triangle mesh to an .obj file (not c binary file)
    outmesh_path = './smpl_template.obj'
    with open( outmesh_path, 'w') as fp:
        for v in m.r:
            fp.write( 'v %f %f %f\n' % ( v[0], v[1], v[2]) )

        for f in m.f+1: # Faces are 1-based, not 0-based in obj files
            fp.write( 'f %d %d %d\n' %  (f[0], f[1], f[2]) )

    ## Print message
    print('..Output mesh saved to: ', outmesh_path) 

if __name__ == "__main__":


    ## 1. Load SMPL model (here we load the female model)
    ## Make sure path is correct
    smpl_path = './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    smpl = load_model(smpl_path)
    
    #### SAVE OBJ ### FOR TEMP 
    #save_to_objfile(smpl)
    #exit()
    
    
    print("pose & beta size:", smpl.pose.size, ",", smpl.betas.size)
    print("pose :", smpl.pose)  # 24*3 = 72 (in 1-D)
    print("shape:", smpl.betas) # 10  (check the latent meaning..)
    ########################
    print("x-range (width): (%5.3f %5.3f) %5.3fm"%(np.min(smpl.r[:,0]), np.max(smpl.r[:,0]), np.max(smpl.r[:,0])- np.min(smpl.r[:,0]))) # x
    print("y-range (tall): (%5.3f %5.3f) %5.3fm"% (np.min(smpl.r[:,1]), np.max(smpl.r[:,1]), np.max(smpl.r[:,1])- np.min(smpl.r[:,1]))) # y
    print("z-range (thick): (%5.3f %5.3f) %5.3fm"%(np.min(smpl.r[:,2]), np.max(smpl.r[:,2]), np.max(smpl.r[:,2])- np.min(smpl.r[:,2]))) # z
  
    
    cam_pose = np.array([
        [1.0,  0.0,  0.0,  0.2],   # zero-rotation  
        [0.0,  1.0,  0.0,  0.0],   # 
        [0.0,  0.0,  1.0,  2.0],   # (0, 0, 2.0) translation  
        [0.0,  0.0,  0.0,  1.0]
    ])
    scene, viewer = setupScene(cam_pose)

    # update pose and shape 
    '''
    m.pose[:] = np.random.rand(m.pose.size) * .2
    m.betas[:] = np.random.rand(m.betas.size) * .03
    '''

    """
    smpl.pose[0:3] = [np.pi/4, np.pi/8, np.pi/4]  # TODO: make straght pose and draw the coordinate figure!
    print("Types:", type(smpl), type(smpl.r), type(smpl.f)) # vertex is variable, f is fixed numpy 
    print("size:", smpl.r.shape, smpl.f.shape)
    """
    
    
    print("make A-pose  ...")
    smpl.pose[:] = 0         
    smpl.pose[16*3 + 2] =  - 7*np.pi/16 # left shoulder, z axis 
    smpl.pose[17*3 + 2] =  + 7*np.pi/16 #
    updateSMPL(scene, viewer, smpl.r, smpl.f) 
    x = input("continue?")    
    
    smpl.betas[1] = -2.0  # fat 
    updateSMPL(scene, viewer, smpl.r, smpl.f) 
    x = input("continue?")    
    
    
    print("The model Rotating around Y axis ....")
    smpl.betas[:] = 0
    smpl.pose[:] = 0
    
    for i in range(32 +1):
        smpl.pose[1] = np.pi*i/16   # 0 joint- y axis
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        time.sleep(0.3)
    
    x = input("continue?")    
    # control left hip joint 
    print("The model lifting left leg by hip joint angle...")
    for i in range(32):
        smpl.pose[1*3 + 0] = -np.pi*i/64   # joint- x axis
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        time.sleep(0.3)
    
   
    x = input("continue?")   
    print("make the model Bigger beta0 ...")
    smpl.pose[:] = 0        
    originalBetas = smpl.betas        
    for i in range(32):
        smpl.betas[0] = i*0.2 # increasing shape param 0  ==> overall size 
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        print("heifht (%5.3f): (%5.3f %5.3f) %5.3fm"% (i*0.2, np.min(smpl.r[:,1]), np.max(smpl.r[:,1]), np.max(smpl.r[:,1])- np.min(smpl.r[:,1]))) # y
        time.sleep(0.3)    

    print("make the model smaller beta0 ...")
    smpl.pose[:] = 0        
    originalBetas = smpl.betas        
    for i in range(32):
        smpl.betas[0] = -i*0.2 # decreasin shape param 0  ==> overall size 
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        print("heifht (%5.3f): (%5.3f %5.3f) %5.3fm"% (-i*0.2, np.min(smpl.r[:,1]), np.max(smpl.r[:,1]), np.max(smpl.r[:,1])- np.min(smpl.r[:,1]))) # y
        time.sleep(0.3)    
        

    x = input("continue?")    
    print("make the model beta1 Thinner ...")  
    smpl.betas[:] = 0         
    for i in range(32):
        smpl.betas[1] =  i*0.1 # increasing shape param 0 ==> thiness 
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        time.sleep(0.3)     

    print("make the model beta1 Fatter ...")
    smpl.betas[1] = 0         
    for i in range(32):
        smpl.betas[1] = -i*0.1 # increasing shape param 0 ==> thiness 
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        time.sleep(0.3)     
   
    
    x = input("continue?")    
    print("make the model beta2 positive")  
    smpl.betas[:] = 0         
    for i in range(32):
        smpl.betas[2] =  i*0.1 # increasing shape param 0 ==> thiness 
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        time.sleep(0.3)     

    print("make the model beta2 negative  ...")
    smpl.betas[:] = 0         
    for i in range(32):
        smpl.betas[2] = -i*0.1 # increasing shape param 0 ==> thiness 
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        time.sleep(0.3)     


    x = input("continue?")    
    print("make the model beta3 positive")  
    smpl.betas[:] = 0         
    for i in range(32):
        smpl.betas[3] =  i*0.1 # increasing shape param 0 ==> thiness 
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        time.sleep(0.3)     

    print("make the model beta3 negative  ...")
    smpl.betas[:] = 0         
    for i in range(32):
        smpl.betas[3] = -i*0.1 # increasing shape param 0 ==> thiness 
        updateSMPL(scene, viewer, smpl.r, smpl.f) 
        time.sleep(0.3)     


  


    x = input("finish?")    
       
    # Close the viwer (and its thread) to finish program     
    viewer.close_external()
    while viewer.is_active:
        pass
               
  
        