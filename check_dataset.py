##   Check the inside of files
##
##  (c) heejune@snut.ac.kr
##    16 Apr 2019
##
##  note: npz file is simply zip (not compressed) file of npy files 
##  
##  10 Oct. 2020: cvt2TPose added
##  
##  

import numpy as np

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import cv2
import time
import sys



###############################################################
# Joint Visualize
###############################################################

# This display setting is required to display the label image in matplot
  
labels = ["background", #     0
            "hat", #            1
            "hair", #           2 
            "sunglass", #       3
            "upper-clothes", #  4
            "skirt",  #          5
            "pants",  #          6
            "dress", #          7
            "belt", #           8
            "left-shoe", #      9
            "right-shoe", #     10
            "face",  #           11
            "left-leg", #       12
            "right-leg", #      13
            "left-arm",#       14
            "right-arm", #      15   
            "bag", #            16
            "scarf" #          17    
        ]  

#https://matplotlib.org/gallery/color/named_colors.html#sphx-glr-gallery-color-named-colors-py
  
label_colors =  ['black', #  "background", #     0
                 'sienna', #"hat", #            1
                 'gray', #"hair", #           2 
                 'navy', #"sunglass", #       3
                 'red',  #"upper-clothes", #  4
                 'gold', #"skirt",  #          5
                 'blue', #"pants",  #          6
                 'seagreen', #"dress", #          7
                 'darkorchid',  #"belt", #           8
                 'firebrick',  #   "left-shoe", #      9
                 'darksalmon', #"right-shoe", #     10
                 'moccasin', #"face",  #           11
                 'darkgreen', #"left-leg", #       12
                 'royalblue', #"right-leg", #      13
                 'chartreuse', #"left-arm",#       14
                 'paleturquoise',  #"right-arm", #      15   
                 'darkcyan', #  "bag", #            16
                 'deepskyblue' #"scarf" #          17    
        ]          
clothnorm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5 ,4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5, 11.5, 12.5, 13.5, 14.5, 15.5, 16.5, 17.5], 18)


lsp_joint_order = [13,  # Head top
                   12,  # Neck
                   8,   # Right shoulder
                   7,   # Right elbow
                   6,   # Right wrist
                   2,   # Right hip
                   1,   # Right knee
                   0,   # Right ankle
                   9,   # Left shoulder
                   10,  # Left elbow
                   11,  # Left wrist
                   3,   # Left hip
                   4,   # Left knee
                   5]   # Left ankle
                   
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
        cv2.circle(img, (x,y), 2, color)
        
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
        
        cv2.line(img, pt1, pt2, color, 1) # 0 : black
     
     
jidx = {"head": 13,  # Head top
            "neck": 12,  # Neck
            "rshl":  8,   # Right shoulder
            "relb":  7,   # Right elbow
            "rwrt":  6,   # Right wrist
            "rhip":  2,   # Right hip
            "rkne": 1,   # Right knee
            "rank": 0,   # Right ankle
            "lshl" : 9,   # Left shoulder
            "lelb": 10,  # Left elbow
            "lwrt": 11,  # Left wrist
            "lhip": 3,   # Left hip
            "lkne": 4,   # Left knee
            "lank": 5}   # Left ankle
            
def cvtTpose(joints):
    
    ''' assume all plannar position 
        if we have 3d joint, we could do better about it
        
        note: joint index  x & y 
              left right   image's human side 
        '''
    tjoints = np.zeros_like(joints)
    
    # head, shoulders, hips 
    for part in ['head', 'neck', 'lshl', 'rshl', 'lhip', 'rhip']:
        tjoints[jidx[part], :] = joints[jidx[part], :]
    
    # elbows, wrist 
    d_r_u1 = np.sqrt(np.sum(np.square(joints[jidx["rshl"], :] - joints[jidx["relb"], :])))
    d_r_u2 = np.sqrt(np.sum(np.square(joints[jidx["relb"], :] - joints[jidx["rwrt"], :])))
    d_l_u1 = np.sqrt(np.sum(np.square(joints[jidx["lshl"], :] - joints[jidx["lelb"], :])))
    d_l_u2 = np.sqrt(np.sum(np.square(joints[jidx["lelb"], :] - joints[jidx["lwrt"], :])))
    tjoints[jidx["relb"], :] = joints[jidx["rshl"],:] - [d_r_u1,0]
    tjoints[jidx["rwrt"], :] = joints[jidx["rshl"],:] - [d_r_u1  + d_r_u2,0]
    tjoints[jidx["lelb"], :] = joints[jidx["lshl"],:] + [d_l_u1,0]
    tjoints[jidx["lwrt"], :] = joints[jidx["lshl"],:] + [d_l_u1 + d_l_u2,0]
    
    # knees and ankle
    d_r_d1 = np.sqrt(np.sum(np.square(joints[jidx["rhip"], :] - joints[jidx["rkne"], :])))
    d_r_d2 = np.sqrt(np.sum(np.square(joints[jidx["rkne"], :] - joints[jidx["rank"], :])))
    d_l_d1 = np.sqrt(np.sum(np.square(joints[jidx["lhip"], :] - joints[jidx["lkne"], :])))
    d_l_d2 = np.sqrt(np.sum(np.square(joints[jidx["lkne"], :] - joints[jidx["lank"], :])))
    tjoints[jidx["rkne"], :] = joints[jidx["rhip"],:] + [0, d_r_d1]
    tjoints[jidx["rank"], :] = joints[jidx["rhip"],:] + [0, d_r_d1 + d_r_d2]
    tjoints[jidx["lkne"], :] = joints[jidx["lhip"],:] + [0, d_l_d1]
    tjoints[jidx["lank"], :] = joints[jidx["lhip"],:] + [0, d_l_d1 + d_l_d2]
    
    return tjoints
    
    

def visualizeJointSegmentation(joints, img_path, seg_path = None):
    ''' visualize one sample '''
    # visualize the pose 
    #joint2d = np.reshape(joints, (-1,2))
    #print('reshaped:', joint2d)    
    img = cv2.imread(img_path)
    if seg_path is not None:
        segimg = cv2.imread(seg_path, 0)
    #print(segimg.shape)
    
    # To check the T-pose test
    bTpose = False
    if bTpose:
        joints = cvtTpose(joints)
    
    drawJoints(img, joints, True)
    drawLimbs(img, joints, True)
  
    if seg_path is not None:
        plt.subplot(1,2,1)
    plt.imshow(img[:,:,::-1]), plt.axis('off'),plt.title('joints')
    if seg_path is not None:
        plt.subplot(1,2,2)
        plt.imshow(segimg), plt.axis('off'),plt.title('segmentation')
    plt.show()

##########################################
# est_joints.npz
##########################################
def visualizeBatchJointSegmentation(jointfile, imgfilePrefix, segfilePrefix = None, debug = False):
    ''' visualize all samples '''

    #print('checking ', jointfile)
    # 1. check joint estimation file format  
    with np.load(jointfile) as zipfile: # zip file loading
        est = zipfile['est_joints']  
        if debug:
            print("shape:", est.shape, ", type:", est.dtype) 
        nimages = 10  # from shape of est or files 
        for imgidx in range(nimages):
            joints = est[:2, :, imgidx].T  # T for joint-wise
            confidence = est[2, :, imgidx]
            if debug:
                print("joints:", joints)
                print("confidence:",confidence)
            imgfile = imgfilePrefix + "%04d"%imgidx + ".jpg" # by search or not
            if segfilePrefix is not None:
                segfile = segfilePrefix + "%04d"%imgidx + ".png"   
            else:
                segfile = None
            if debug:
                print(imgfile, segfile)
            visualizeJointSegmentation(joints, imgfile, segfile)

'''    
def temp( ):

    segimg = cv2.imread('./dataset/10k/parsed/dataset10k_0000.png', 0)
    resized = cv2.resize(segimg, (400,600), interpolation = cv2.INTER_NEAREST )
    cv2.imwrite('r.png', resized)
'''

def test_10k():

    # default params 
    joint_file_path  = './dataset/10k/est_joints.npz'
    imge_file_prefix =  './dataset/10k/input/dataset10k_'
    segmentation_file_prefix = "./dataset/10k/parsed/dataset10k_"
    '''
    if len(sys.argv) > 3:
        joint_file_path = sys.argv[1]
        imge_file_prefix = sys.argv[2]
        segmentation_file_prefix = sys.argv[3]
    '''    
    # run batch processing
    visualizeBatchJointSegmentation(joint_file_path, imge_file_prefix, segmentation_file_prefix)
   

def test_beforeafter():

    # default params 
    joint_file_path  = './dataset/beforeafter/est_joints.npz'
    imge_file_prefix =  './dataset/beforeafter/input/img_'
 
    # run batch processing
    visualizeBatchJointSegmentation(joint_file_path, imge_file_prefix, None)
       
    
if __name__ == '__main__':


    #test_10k()
    test_beforeafter()
    
   