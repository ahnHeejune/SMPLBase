###################################################################
# OpenDr Python 3.x check 
###################################################################
# only camera 
# 
# (c) 2020  heejune@seoultech.ac.kr
###################################################################


from opendr import camera
import numpy as np
import chumpy as ch
import matplotlib.pyplot as plt
import cv2
from smpl.serialization import load_model  
import graphutil

"""
         OBJECT
           | -2
           |   
           |
    ------(x)---------------> x
           |
           |
           |
           v 
           z 
           


"""

def camera_test_points(): 

    # scene setting 
    
    v3d = np.sin(np.arange(30)).reshape((-1,3))  # vertices: 10x3
    v3d[:, 2] -= 2            # z axis
    print(type(v3d), v3d.shape)
    print(v3d)        
            
    rt = ch.zeros(3)            # rotation angles 
    t = ch.zeros(3)             # translation vector 
    f = ch.array([250,250])     # focal length  (fx, fy)
    c = ch.array([200,300])     # center offset (x, y)
    k = ch.zeros(5)             # ?    

    
    cam = camera.ProjectPoints(f=f, rt=rt, t=t, k=k, c=c)
    #print('cam:', inspect.getmro(ProjectPoints))

    # projection 
    cam.v = v3d
    print(type(cam.r), cam.r.shape)
    print(cam.r)
    
    
    #####
    w = 400
    h = 600
    img = np.zeros((h,w,3))  # float ?
    # mapping points into the 
    for i in range(cam.r.shape[0]):
        x, y = int(cam.r[i, 0]), int(cam.r[i,1])
        print(x, y)
        if x >= 0 and x < w and y >= 0 and y < h:
            #img[y,x, :] = 1.0
            cv2.circle(img, (x,y), 5, (255,255,255), -1)
  
    depth = graphutil.build_depthmap2(v3d, cam)  ## todo 
    uvd  = np.zeros(v3d.shape)
    uvd[:,0] = cam.r[:,0] #
    uvd[:,1] = cam.r[:,1] # 
    uvd[:,2] = depth   #  
    recon_v3d = cam.unproject_points(uvd)
    print(type(recon_v3d), recon_v3d.shape)
    print(recon_v3d)
    
    
    plt.imshow(img), plt.axis('off')
    plt.show()
    


def camera_test_smpl(): 

    # 1. scene setting 
    smpl_path = './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    smpl = load_model(smpl_path)
    v3d = smpl.r 
    print(type(v3d), v3d.shape)
    #print(v3d)        
          
    # 2. camera setup 
    # pysical 
    rt = ch.zeros(3)            # rotation angles    (0,0,0)
    t = ch.array([0,0, 1.5])      # translation vector (0,0,1)
    
    # sensor-imaging 
    f = ch.array([250,250])     # focal length  (fx, fy)
    c = ch.array([200,300])     # center offset (x, y)
    k = ch.zeros(5)             #     

    # projection 
    cam = camera.ProjectPoints(f=f, rt=rt, t=t, k=k, c=c)
    #print('cam:', inspect.getmro(ProjectPoints))

    # projection 
    cam.v = v3d
    print(type(cam.r), cam.r.shape)
    #print(cam.r)
    
    
    #####
    w = 400
    h = 600
    img = np.zeros((h,w,3))  # float ?
    # mapping points into the 
    for i in range(cam.r.shape[0]):
        x, y = int(cam.r[i, 0]), int(cam.r[i,1])
        if x >= 0 and x < w and y >= 0 and y < h:
            img[y, x, :] = 1.0 
  
    # reconstruction 
    depth = graphutil.build_depthmap2(v3d, cam)  ## todo 
    uvd  = np.zeros(v3d.shape)
    uvd[:,0] = cam.r[:,0] #
    uvd[:,1] = cam.r[:,1] # 
    uvd[:,2] = depth   #  
    recon_v3d = cam.unproject_points(uvd)
    print(type(recon_v3d), recon_v3d.shape)
    #print(recon_v3d)
    
    
    plt.imshow(img[::-1,:,:]), plt.axis('off')  # y-axis upside-down
    plt.show() 
 

if __name__ == "__main__":
    
    # points test     
    # camera_test_points() 
    
    camera_test_smpl() 
