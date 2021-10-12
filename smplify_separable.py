"""

Based on smplify_pyrender.py

Modification : heejune@seoultech.ac.kr 
- For Windows Enviroment, we use only Camera module from OpenDR and replay the Rendering with PyRedner
- Shape and Pose separable (first fit the shape in T-Pose and then apply Pose)
==============

dependent 
===============

numpy    (pip install numpy) 
opencv   (pip install opencv-python)
chumpy   (pip install chumpy)
smpl     (modified by seoultech)
opendr   (modified by SeoulTech) 
smplify  (named for package) 


"""

from os.path import join, exists, abspath, dirname
from os import makedirs
import logging
#import cPickle as pickle   # Python 2
import _pickle as pickle   # Python 3
from time import time
from glob import glob
import argparse

import inspect  # for debugging

# basics 
import cv2
import numpy as np
import chumpy as ch

# SMPL 
#from smpl_webuser.serialization import load_model
from smpl.serialization import load_model
#from smpl_webuser.lbs import global_rigid_transformation
from smpl.lbs import global_rigid_transformation
#from smpl_webuser.verts import verts_decorated
from smpl.verts import verts_decorated

# OPENDR
from opendr.camera import ProjectPoints  

# SMPLIFY
#from lib.robustifiers import GMOf
from smplify.robustifiers import GMOf
#from lib.sphere_collisions import SphereCollisions
from smplify.sphere_collisions import SphereCollisions
#from lib.max_mixture_prior import MaxMixtureCompletePrior
from smplify.max_mixture_prior import MaxMixtureCompletePrior

# Pyrender
from pyrender_model import render_model  #


_LOGGER = logging.getLogger(__name__)

# Mapping from LSP joints to SMPL joints.
# 0 Right ankle  8
# 1 Right knee   5
# 2 Right hip    2
# 3 Left hip     1
# 4 Left knee    4
# 5 Left ankle   7
# 6 Right wrist  21
# 7 Right elbow  19
# 8 Right shoulder 17
# 9 Left shoulder  16
# 10 Left elbow    18
# 11 Left wrist    20
# 12 Neck           -
# 13 Head top       added

test_img_ind = 27

def cvtTpose(joints):
    
    ''' assume all plannar position 
        if we have 3d joint, we could do better about it
        
        note: joint index  x & y 
              left right   image's human side 
    '''
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

# --------------------Camera estimation --------------------
def guess_init(model, focal_length, j2d, init_pose):
    """Initialize the camera translation via triangle similarity, by using the torso joints        .
    :param model: SMPL model
    :param focal_length: camera focal length (kept fixed)
    :param j2d: 14x2 array of CNN joints
    :param init_pose: 72D vector of pose parameters used for initialization (kept fixed)
    :returns: 3D vector corresponding to the estimated camera translation
    """
    cids = np.arange(0, 12)
    # map from LSP to SMPL joints
    j2d_here = j2d[cids]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])
    Jtr = Jtr[smpl_ids].r

    # 9 is L shoulder, 3 is L hip
    # 8 is R shoulder, 2 is R hip
    diff3d = np.array([Jtr[9] - Jtr[3], Jtr[8] - Jtr[2]])
    mean_height3d = np.mean(np.sqrt(np.sum(diff3d**2, axis=1)))

    diff2d = np.array([j2d_here[9] - j2d_here[3], j2d_here[8] - j2d_here[2]])
    mean_height2d = np.mean(np.sqrt(np.sum(diff2d**2, axis=1)))

    est_d = focal_length * (mean_height3d / mean_height2d)
    # just set the z value
    init_t = np.array([0., 0., est_d])
    return init_t


def initialize_camera(model,
                      j2d,
                      img,
                      init_pose,
                      flength=5000.,
                      pix_thsh=25.,
                      viz=False):
    """Initialize camera translation and body orientation
    :param model: SMPL model
    :param j2d: 14x2 array of CNN joints
    :param img: h x w x 3 image 
    :param init_pose: 72D vector of pose parameters used for initialization
    :param flength: camera focal length (kept fixed)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing the estimated camera,
              a boolean deciding if both the optimized body orientation and its flip should be considered,
              3D vector for the body orientation
    """
    # optimize camera translation and body orientation based on torso joints
    # LSP torso ids:
    # 2=right hip, 3=left hip, 8=right shoulder, 9=left shoulder
    torso_cids = [2, 3, 8, 9]
    # corresponding SMPL torso ids
    torso_smpl_ids = [2, 1, 17, 16]

    center = np.array([img.shape[1] / 2, img.shape[0] / 2])

    # initialize camera rotation
    rt = ch.zeros(3)
    # initialize camera translation
    _LOGGER.info('initializing translation via similar triangles')
    init_t = guess_init(model, flength, j2d, init_pose)
    t = ch.array(init_t)

    # check how close the shoulder joints are
    try_both_orient = np.linalg.norm(j2d[8] - j2d[9]) < pix_thsh

    opt_pose = ch.array(init_pose)
    (_, A_global) = global_rigid_transformation(
        opt_pose, model.J, model.kintree_table, xp=ch)
    Jtr = ch.vstack([g[:3, 3] for g in A_global])

    # initialize the camera
    cam = ProjectPoints(
        f=np.array([flength, flength]), rt=rt, t=t, k=np.zeros(5), c=center)
    print('cam:', inspect.getmro(ProjectPoints))

    
    # we are going to project the SMPL joints
    cam.v = Jtr

    if viz:
    
        #print("=============================>>> registering onstep")
        viz_img = img.copy()

        # draw thetry_both_orient target (CNN) joints
        for coord in np.around(j2d).astype(int):
            if (coord[0] < img.shape[1] and coord[0] >= 0 and
                    coord[1] < img.shape[0] and coord[1] >= 0):
                cv2.circle(viz_img, tuple(coord), 3, [0, 255, 0])

        import matplotlib.pyplot as plt
        plt.ion()

        # draw optimized joints at each iteration
        def on_step(_):
            """Draw a visualization."""
            #print("entered on_step()")
            fig = plt.figure(1, figsize=(7, 7))
            plt.subplot(1, 2, 1)  # do we need this?
            
            viz_img = img.copy()
            for coord in np.around(cam.r[torso_smpl_ids]).astype(int):
                if (coord[0] < viz_img.shape[1] and coord[0] >= 0 and
                        coord[1] < viz_img.shape[0] and coord[1] >= 0):
                    cv2.circle(viz_img, tuple(coord), 3, [0, 0, 255])
            plt.imshow(viz_img[:, :, ::-1])
            plt.draw()
            plt.show()
            plt.pause(1e-3)
    else:
        on_step = None
        
        
    # optimize for camera translation and body orientation
    free_variables = [cam.t, opt_pose[:3]]
    ch.minimize(
        # data term defined over torso joints...
        {'cam': j2d[torso_cids] - cam[torso_smpl_ids],
         # ...plus a regularizer for the camera translation
         'cam_t': 1e2 * (cam.t[2] - init_t[2])},
        x0=free_variables,
        method='dogleg',
        callback=on_step,
        options={'maxiter': 100,
                 'e_3':  0.01,  #.0001,
                 # disp set to 1 enables verbose output from the optimizer
                 'disp': 0})
    if viz:
        plt.ioff()
    return (cam, try_both_orient, opt_pose[:3].r)


# --------------------Core optimization --------------------
def optimize_on_joints(j2d,
                       model,
                       cam,
                       img,
                       prior,
                       try_both_orient,
                       body_orient,
                       n_betas=10,
                       regs=None,
                       conf=None,
                       viz=False,
                       save_path=None):
    """Fit the model to the given set of joints, given the estimated camera
    :param j2d: 14x2 array of CNN joints
    :param model: SMPL model
    :param cam: estimated camera
    :param img: h x w x 3 image 
    :param prior: mixture of gaussians pose prior
    :param try_both_orient: boolean, if True both body_orient and its flip are considered for the fit
    :param body_orient: 3D vector, initialization for the body orientation
    :param n_betas: number of shape coefficients considered during optimization
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param conf: 14D vector storing the confidence values from the CNN
    :param viz: boolean, if True enables visualization during optimization
    :returns: a tuple containing the optimized model, its joints projected on image space, the camera translation
    """
    t0 = time()
    # define the mapping LSP joints -> SMPL joints
    # cids are joints ids for LSP:
    cids = list(range(12)) + [13]
    #cids = [0,1,2,3,4,5,6,7,8,9,10,11,13]
    
    # joint ids for SMPL
    # SMPL does not have a joint for head, instead we use a vertex for the head
    # and append it later.
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]

    # the vertex id for the joint corresponding to the head
    head_id = 411

    # weights assigned to each joint during optimization;
    # the definition of hips in SMPL and LSP is significantly different so set
    # their weights to zero
    base_weights = np.array(
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)  # no LSP

    #    [1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float64)

    if try_both_orient:
        flipped_orient = cv2.Rodrigues(body_orient)[0].dot(
            cv2.Rodrigues(np.array([0., np.pi, 0]))[0])
        flipped_orient = cv2.Rodrigues(flipped_orient)[0].ravel()
        orientations = [body_orient, flipped_orient]
    else:
        orientations = [body_orient]

    if try_both_orient:
        # store here the final error for both orientations,
        # and pick the orientaoptimize_on_jointstion resulting in the lowest error
        errors = []

    svs = []
    cams = []
    ##print('orient:', orientations)
    ##print('prior.weights:', prior.weights) # N
    ##print('prior.means:', prior.means)  # N*23?
    #print('prior.means.shape:', prior.means,shape)
    #print('n_betas:', n_betas)
    for o_id, orient in enumerate(orientations):
        # initialize the shape to the mean shape in the SMPL training set
        betas = ch.zeros(n_betas)
        #betas[0] = 3.0 ########## from height input?

        # initialize the pose by using the optimized body orientation and the
        # pose prior
        t = np.zeros_like(prior.weights.dot(prior.means))
        #init_pose = np.hstack((orient, prior.weights.dot(prior.means)))
        init_pose = np.hstack((orient, t)) # T-pose 

        # instantiate the model:
        # verts_decorated allows us to define how many
        # shape coefficients (directions) we want to consider (here, n_betas)
        sv = verts_decorated(
            trans=ch.zeros(3),
            pose=ch.array(init_pose),
            v_template=model.v_template,
            J=model.J_regressor,
            betas=betas,
            shapedirs=model.shapedirs[:, :, :n_betas],
            weights=model.weights,
            kintree_table=model.kintree_table,
            bs_style=model.bs_style,
            f=model.f,
            bs_type=model.bs_type,
            posedirs=model.posedirs)

        # make the SMPL joints depend on betas
        Jdirs = np.dstack([model.J_regressor.dot(model.shapedirs[:, :, i])
                           for i in range(len(betas))])
        J_onbetas = ch.array(Jdirs).dot(betas) + model.J_regressor.dot(
            model.v_template.r)

        # get joint positions as a function of model pose, betas and trans
        (_, A_global) = global_rigid_transformation(
            sv.pose, J_onbetas, model.kintree_table, xp=ch)
        Jtr = ch.vstack([g[:3, 3] for g in A_global]) + sv.trans

        # add the head joint, corresponding to a vertex...
        Jtr = ch.vstack((Jtr, sv[head_id]))

        # ... and add the joint id to the list
        if o_id == 0:
            smpl_ids.append(len(Jtr) - 1)

        # update the weights using confidence values
        weights = base_weights * conf[
            cids] if conf is not None else base_weights

        # project SMPL joints on the image plane using the estimated camera
        cam.v = Jtr

        # obj1. data term: distance between observed and estimated joints in 2D
        ###########################
        #obj_j2d = lambda w, sigma: (
        #    w * weights.reshape((-1, 1)) * GMOf((j2d[cids] - cam[smpl_ids]), sigma))

        # obj2: mixture of gaussians pose prior
        ###########################
        pprior = lambda w: w * prior(sv.pose)
        # obj3: joint angles pose prior, defined over a subset of pose parameters:
        # 55: left elbow,  90deg bend at -np.pi/2
        # 58: right elbow, 90deg bend at np.pi/2
        # 12: left knee,   90deg bend at np.pi/2
        # 15: right knee,  90deg bend at np.pi/2
        alpha = 10
        my_exp = lambda x: alpha * ch.exp(x)
        obj_angle = lambda w: w * ch.concatenate([my_exp(sv.pose[55]), my_exp(-sv.pose[
                                                 58]), my_exp(-sv.pose[12]), my_exp(-sv.pose[15])])

        if viz:
            import matplotlib.pyplot as plt
            plt.ion()
            global itern
            global stagen
            stagen = 0
            itern = 0
            #global ax1, ax2
            #ax1 = None
            #ax2 = None
           
            # Call back When one Optimization is done #
            
            def on_step(_):  
            
                global itern #, ax1, ax2 	
                itern = itern + 1
               
                # Create visualization.
                fig = plt.figure(1, figsize=(7, 7))
                
                # show optimized joints in 2D
                tmp_img = img.copy()  # not to modify the original image 
                joint_loc = 0
                marker_scale = 2
                for coord, target_coord in zip(np.around(cam.r[smpl_ids]).astype(int),np.around(j2d[cids]).astype(int)):
                    if (coord[0] < tmp_img.shape[1] and coord[0] >= 0 and
                            coord[1] < tmp_img.shape[0] and coord[1] >= 0):
                        #cv2.circle(tmp_img, tuple(coord), 3, [0, 0, 255])
                        cv2.drawMarker(tmp_img, tuple(coord), [0, 0, 255], markerSize = 5*marker_scale)
  
                    if (target_coord[0] < tmp_img.shape[1] and
                            target_coord[0] >= 0 and
                            target_coord[1] < tmp_img.shape[0] and
                            target_coord[1] >= 0):
                        if joint_loc in [3,4,5,9,10,11]: # LEFT
                       	    cv2.circle(tmp_img, tuple(target_coord), 3*marker_scale,
                                   [0, 255, 0]) # Green
                        else:
                       	    cv2.circle(tmp_img, tuple(target_coord), 3*marker_scale,
                                   [255, 0, 0]) # Blue
                    joint_loc = joint_loc + 1
 
                #if ax1 is None:
                plt.subplot(1, 2, 1)
                #fig.subplot(1, 2, 1)
                plt.imshow(tmp_img[:, :, ::-1]), plt.axis('off'), plt.title('input & est. joints')  #plt.draw()
                #plt.subplot(122) #plt.cla()
                #axis1.cla()        
                h, w = img.shape[0:2]  
                dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])
     
                rendered = (render_model(
                    sv.r, model.f, w, h, cam, far=20 + dist, img =img[:,:,::-1]) * 255.).astype('uint8') # background 
                
                plt.subplot(1, 2, 2)
                plt.imshow(rendered), plt.title('st=' + str(stagen) +' itr=' + str(itern)), plt.axis('off')
                plt.show()
                
                # save input and result together 
                if save_path is not None:
                    _LOGGER.info("Saving step image:" + save_path + '_' + str(itern) + '.png')
                    plt.savefig(save_path + '_' + str(itern) + '.png')
                    #global test_img_ind
                    #plt.savefig('smplify2d3d_' + str(test_img_ind) + '_' + str(itern) + '.png')

                if 'objs' in locals():
                    for k in objs.keys():
                        print("error-%s:%.1f"%(k, np.sum(objs[k].r**2)))                
                print('betas:', sv.betas.r)
                    
                plt.pause(1e-2)

            on_step(_) # call at init 
        else:
            on_step = None

        # obj5: interpenentration
        ###########################
        if regs is not None:
            # interpenetration term
            sp = SphereCollisions(
                pose=sv.pose, betas=sv.betas, model=model, regs=regs)
            sp.no_hands = True

        #############################################
        # 5. optimize
        #############################################
        # weight configuration used in the paper, with joints + confidence values from the CNN
        # (all the weights used in the code were obtained via grid search, see the paper for more details)
        # the first list contains the weights for the pose priors,
        # the second list contains the weights for the shape prior
        
        ##########################################
        # 5.1 Shape estimation in T-Pose 
        ##########################################
        j2d_TPose  = cvtTpose(j2d)
        obj_j2d = lambda w, sigma: (
            w * weights.reshape((-1, 1)) * GMOf((j2d_TPose[cids] - cam[smpl_ids]), sigma))

        #opt_weights = zip([4.04 * 1e2, 4.04 * 1e2, 57.4, 4.78],
        #                    [1e2, 5 * 1e1, 1e1, .5 * 1e1])      
        opt_weights = zip([4.4, 0.78],  # small priors ! 
                          [.5 * 1e1, .2 * 1e1]) # beta: too small => arbitrary shape
        #print('opt-weights:', opt_weights)
        # run the optimization in 4 stages, progressively decreasing the
        # weights for the priors
        for stage, (w, wbetas) in enumerate(opt_weights):
            stagen = stage
            _LOGGER.info('stage %01d, total itern %d'%(stage, itern))
            objs = {}
            objs['j2d'] = obj_j2d(1., 100)
            #objs['pose'] = pprior(w)
            #objs['pose_exp'] = obj_angle(0.317 * w)
            objs['betas'] = wbetas * betas
            #if regs is not None:
            #    objs['sph_coll'] = 1e3 * sp

            free_variables = [sv.betas] # [sv.betas, sv.pose]
            b  = ch.minimize(
                       objs,              # objective functions
                       x0=free_variables, # free-variables
                       method='dogleg',
                       callback=on_step,
                       options={'maxiter': 10, 'e_3': .001, 'disp': 0})
            print('stage-', stage, ' done')
            print('   betas;', sv.betas.r)
            #print('   pose :', p.reshape((-1,3)))
       
       
       
        ##############################
        # after setting the pose 
        ##############################
        #sv.pose[:] = np.hstack((orient, prior.weights.dot(prior.means)))  # TODO: use input joint to pose angle!!
        sv.pose[16*3 + 2] =  - 7*np.pi/16 # left shoulder, z axis 
        sv.pose[17*3 + 2] =  + 7*np.pi/16 # right shoulder, z axis 
        
        obj_j2d = lambda w, sigma: (
            w * weights.reshape((-1, 1)) * GMOf((j2d[cids] - cam[smpl_ids]), sigma))

        opt_weights = zip([4.4, 4.4, 4.4, 0.78],  # small priors ! 
                          [.5 * 1e1, .5 * 1e1, .5 * 1e1, .5 * 1e1]) # beta: too small => arbitrary shape
        #print('opt-weights:', opt_weights)
        # run the optimization in 4 stages, progressively decreasing the
        # weights for the priors
        for stage, (w, wbetas) in enumerate(opt_weights):
            stagen = stage
            _LOGGER.info('stage %01d, total itern %d'%(stage, itern))
            objs = {}
            objs['j2d'] = obj_j2d(1., 100)
            objs['pose'] = pprior(w)
            objs['pose_exp'] = obj_angle(0.317 * w)
            objs['betas'] = wbetas * betas
            if regs is not None:
                objs['sph_coll'] = 1e3 * sp

            b, p = ch.minimize(
                       objs,                   # objective functions
                       x0=[sv.betas, sv.pose], # free-variables
                       method='dogleg',
                       callback=on_step,
                       options={'maxiter': 10, 'e_3': .01, 'disp': 0})
            print('stage-', stage, ' done')
            print('   betas;', sv.betas.r)
            #print('   pose :', p.reshape((-1,3)))
       
            # Intermediate step figures ############################
            """
            import matplotlib.pyplot as plt
            plt.ion()
            plt.show()
            # 1. 2-D input 
            plt.subplot(121)	
            plt.imshow(img[:, :, ::-1]) # It will be nicer if we have marker here
            # 2 rendered 3-D to 2D
            plt.subplot(122)
            plt.cla()           
            rendered = (render_model(
                 verts, model.f, w, h, cam, far=20 + dist) * 255.).astype('uint8')
            plt.imshow(rendered)
            plt.draw()
            # save input and result together 
            plt.savefig(out_path.replace('.pkl', '.png'))
            """
            ######################################################

        t1 = time()
        _LOGGER.info('elapsed %.05f', (t1 - t0))
        if try_both_orient:
            errors.append((objs['j2d'].r**2).sum())
        svs.append(sv)
        cams.append(cam)

    if try_both_orient and errors[0] > errors[1]:
        choose_id = 1
    else:
        choose_id = 0
    if viz:
        plt.ioff()
    return (svs[choose_id], cams[choose_id].r, cams[choose_id].t.r)




""" 

   Sinle Image Fitting Function  

"""
def run_single_fit(img,
                   j2d,
                   conf,
                   model,
                   regs=None,
                   n_betas=10,
                   flength=5000.,
                   pix_thsh=25.,
                   scale_factor=1,
                   viz=False, #optimize_on_joints
                   do_degrees=None,
                   save_path=None):
    """Run the fit for one specific image.
    :param img: h x w x 3 image 
    :param j2d: 14x2 array of CNN joints
    :param conf: 14D vector storing the confidence values from the CNN
    :param model: SMPL model
    :param regs: regressors for capsules' axis and radius, if not None enables the interpenetration error term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (kept fixed during optimization)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param scale_factor: int, rescale the image (for LSP, slightly greater images -- 2x -- help obtain better fits)
    :param viz: boolean, if True enables visualization during optimization
    :param do_degrees: list of degrees in azimuth to render the final fit when saving results
    :returns: a tuple containing camera/model parameters and images with rendered fits
    """

    #print('>>> model>>>>>>>>>>>>>>>>>>>>>')
    #print(dir(model))
    '''
    print('kintree_table', model.kintree_table)
    print('J:', model.J)
    print('v_template:', model.v_template)
    print('J_regressor:', model.J_regressor)
    print('shapedirs:', model.shapedirs)
    print('weights:', model.weoptimize_on_jointsights)
    print('bs_style:', model.bs_style)
    print('f:', model.f)
    '''

    if do_degrees is None:
        do_degrees = []
 
    ###################################
    # 1. prior setting
    ###################################
    # create the pose prior (GMM over CMU)
    prior = MaxMixtureCompletePrior(n_gaussians=8).get_gmm_prior()
    # get the mean pose as our initial pose
    init_pose = np.hstack((np.zeros(3), prior.weights.dot(prior.means)))
    print(' gmm: w=', prior.weights.r)   # pose GMM statistics chumpy
    print(' gmm: m=', prior.means.shape) # pose GMM statistics numpy
    print(" inital pose:", init_pose.reshape((-1,3)))  # numpy  

    if scale_factor != 1:
        img = cv2.resize(img, (img.shape[1] * scale_factor,
                               img.shape[0] * scale_factor))
        j2d[:, 0] *= scale_factor
        j2d[:, 1] *= scale_factor

    ###################################
    # 2. guess the camera parameters
    ###################################
    (cam, try_both_orient, body_orient) = initialize_camera(
        model,
        j2d,
        img,
        init_pose,
        flength=flength,
        pix_thsh=pix_thsh,
        viz=viz)
    _LOGGER.info('Camera init done======================')

    ###################################
    # 3. fit
    ####################################
    (sv, opt_j2d, t) = optimize_on_joints(
        j2d,     # 2-D joints coordinate, scaled
        model,   # 
        cam,     # camera param
        img,     # 2-d img scaled
        prior,# priors
        try_both_orient, # frontal and back
        body_orient, #
        n_betas=n_betas,  # shape params size
        conf=conf,   # confidence of 2-d joints
        viz=viz,     # visualizing or not
        regs=regs, 
        save_path=save_path 
        ) # 

    h = img.shape[0]
    w = img.shape[1]
    dist = np.abs(cam.t.r[2] - np.mean(sv.r, axis=0)[2])
    ####################################
    # 4. rendering to different view angles
    ####################################
    images = []
    orig_v = sv.r
    for deg in do_degrees:
        if deg != 0:
            aroundy = cv2.Rodrigues(np.array([0, np.radians(deg), 0]))[0]
            center = orig_v.mean(axis=0)
            new_v = np.dot((orig_v - center), aroundy)
            verts = new_v + center
        else:
            verts = orig_v
        # now render with background of original input image
        im = (render_model(
            verts, model.f, w, h, cam, far=20 + dist, img=img[:,:,::-1]) * 255.).astype('uint8')
        images.append(im)

    # **. annotation to the original image 
    cids = list(range(12)) + [13]
    smpl_ids = [8, 5, 2, 1, 4, 7, 21, 19, 17, 16, 18, 20]
    joint_loc = -1
    marker_scale = 2
    for coord, target_coord in zip(
                        np.around(cam.r[smpl_ids]).astype(int),
                        np.around(j2d[cids]).astype(int)):
        joint_loc = joint_loc + 1
        if (coord[0] < img.shape[1] and coord[0] >= 0 and
                            coord[1] < img.shape[0] and coord[1] >= 0):
            #cv2.circle(tmp_img, tuple(coord), 3, [0, 0, 255])
            cv2.drawMarker(img, tuple(coord), [0, 0, 255], markerSize = 5*marker_scale)
  
            if (target_coord[0] < img.shape[1] and
                target_coord[0] >= 0 and
                target_coord[1] < img.shape[0] and
                target_coord[1] >= 0):
                if joint_loc in [3,4,5,9,10,11]: # LEFT
                    cv2.circle(img, tuple(target_coord), 3*marker_scale, [0, 255, 0]) # Green
                else:
                     cv2.circle(img, tuple(target_coord), 3*marker_scale,[255, 0, 0]) # Blue
          
   
    # 5. return resultant fit parameters  (pose, shape) is what we want but camera needed
    ''' # not eough for re-rendering
    params = {'cam_t': cam.t.r,
              'f': cam.f.r,
              'pose': sv.pose.r,
              'betas': sv.betas.r}
    '''
    # save all Camera parameters and SMPL paramters 
    params = {'cam_f': cam.f.r,
              'cam_rt': cam.rt.r,
              'cam_t': cam.t.r,
              'cam_k': cam.k.r,
              'cam_c': cam.c.r,
              'pose': sv.pose.r,
              'betas': sv.betas.r}

    return params, images


"""

  main 

"""

def main(base_dir,
         out_dir,
         use_interpenetration=True,
         n_betas=10,
         flength=5000.,
         pix_thsh=25.,
         gender= 'neutral',  # male, female, neutral 
         #use_neutral=False,
         viz=True,
         dataset='lsp',
         scale = 2,
         savestep = False):

    """Set up dataset dependent paths to image and joint data, saves results.

    :param base_dir: folder containing LSP images and data
    :param out_dir: output folder
    :param use_interpenetration: boolean, if True enables the interpenetration term
    :param n_betas: number of shape coefficients considered during optimization
    :param flength: camera focal length (an estimate)
    :param pix_thsh: threshold (in pixel), if the distance between shoulder joints in 2D
                     is lower than pix_thsh, the body orientation as ambiguous (so a fit is run on both
                     the estimated one and its flip)
    :param use_neutral: boolean, if True enables uses the neutral gender SMPL model
    :param viz: boolean, if True enables visualization during optimization

    dataset:
    scale  :
    """

    img_dir = join(abspath(base_dir), 'input')
    data_dir = join(abspath(base_dir),'results')

    if not exists(out_dir):
        makedirs(out_dir)

    #if not exists(out_dir +'/step'):
    #    makedirs(out_dir +'/step')


    # Render degrees: List of degrees in azimuth to render the final fit.
    # Note that rendering many views can take a while.
    do_degrees = [0.]

    sph_regs = None
    '''
    if not use_neutral:
        _LOGGER.info("Reading genders...")
        # File storing information about gender in LSP
        with open(join(data_dir, dataset + '_gender.csv')) as f:
            genders = f.readlines()
        model_female = load_model(MODEL_FEMALE_PATH)
        model_male = load_model(MODEL_MALE_PATH)
        if use_interpenetration:
            sph_regs_male = np.load(SPH_REGS_MALE_PATH)
            sph_regs_female = np.load(SPH_REGS_FEMALE_PATH)
    else:
        gender = 'neutral'
        model = load_model(MODEL_NEUTRAL_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_NEUTRAL_PATH)
    '''
    if gender == 'male':
        model_male = load_model(MODEL_MALE_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_MALE_PATH)
    elif gender == 'female':
        model = load_model(MODEL_FEMALE_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_FEMALE_PATH)
    else:
        gender == 'neutral'
        model = load_model(MODEL_NEUTRAL_PATH)
        if use_interpenetration:
            sph_regs = np.load(SPH_REGS_NEUTRAL_PATH)
   

    _LOGGER.info("Reading genders done")
    _LOGGER.info("Loading joints ...")

    # Load joints
    est = np.load(join(base_dir, 'est_joints.npz'))['est_joints']
    #print('est_shape:', est.shape)
    _LOGGER.info("Loading joints done.")

    # Load images
    img_paths = sorted(glob(join(img_dir, '*[0-9].jpg')))
    
    # 3. SMPLify  all images 
    for ind, img_path in enumerate(img_paths):

        _LOGGER.info('processing input:' + img_path)

        out_path = '%s/%04d.pkl' % (out_dir, ind)  # prefix and  out_param, out_img is better name
        if not exists(out_path):
        #global test_img_ind
        #if ind == test_img_ind:  # for testing    
            _LOGGER.info('Fitting 3D body on `%s` (saving to `%s`).', img_path, out_path)
            
            if not exists(img_path):
                print("no file", img_path)
            img = cv2.imread(img_path)
            if img.ndim == 2:
                _LOGGER.warn("The image is grayscale!")
                img = np.dstack((img, img, img))

            joints = est[:2, :, ind].T
            conf = est[2, :, ind]

            '''
            #gender = 'male' if int(genders[ind]) == 0 else 'female'
            if gender == 'female':
                model = model_female
                if use_interpenetration:
                    sph_regs = sph_regs_female
            elif gender == 'male':
                model = model_male
                if use_interpenetration:
                    sph_regs = sph_regs_male
            '''
            
            #print('model:',model.show_tree())
            # have to see inside !!!!!!
            #model.show_tree()
            #print('sph:', sph_regs)
            #print('sph.shape:', sph_regs.shape)

            
            step_path = '%s/step/%04d' % (out_dir, ind) if savestep else None
            #print('step_path:', step_path)

            # paramters and projections (viz)
            params, vis = run_single_fit(
                img,
                joints,
                conf,
                model,
                regs=sph_regs,
                n_betas=n_betas,
                flength=flength,
                pix_thsh=pix_thsh,
                scale_factor= scale, #2,
                viz=viz,
                do_degrees=do_degrees,
                save_path=step_path)

            #print('>>>>>>>final optimization result')
            #print('\t3-d param:', params)
            #print('\tlen(vis):', len(vis))
            #print('\ttype(vis[0]):', type(vis[0]))
            #print('\tvis[0],shape:', vis[0].shape)

            if viz:
                import matplotlib.pyplot as plt
                plt.ion()
                plt.show()
                num_images = 1
                if do_degrees is not None
                    num_images = len(do_degrees) + 1
                plt.subplot(1, num_images, 1)	
                plt.imshow(img[:, :, ::-1]) # It will be nicer if we have marker here
                if do_degrees is not None:
                    for di, deg in enumerate(do_degrees):
                        plt.subplot(1, num_images, di +1)
                        plt.cla()
                        plt.imshow(vis[di])
                        plt.draw()
                        plt.title('%d deg' % deg)

                    
                #raw_input('Press any key to continue...')

            #print(params)
            # camera, camera f, pose and shape for reuse (can we? render.py?)
            with open(out_path, 'wb') as outf:  # 'wb' for python 3?
                pickle.dump(params, outf)

            # This only saves the first rendering.
            if do_degrees is not None:
                # SMPL only?
                #cv2.imwrite(out_path.replace('.pkl', '.png'), vis[0])
                # save input and result together (all degrees )
                import matplotlib.pyplot as plt
                plt.ion()
                plt.show()
                '''plt.subplot(121)	
                plt.imshow(img[:, :, ::-1]) # It will be nicer if we have marker here
                plt.subplot(122)
                '''
                num_images = len(do_degrees) + 1
                plt.subplot(1, num_images, 1)	
                plt.imshow(img[:, :, ::-1]) # It will be nicer if we have marker here
                if do_degrees is not None:
                    for di, deg in enumerate(do_degrees):
                        plt.subplot(1, num_images, di +1)
                        plt.cla()
                        plt.imshow(vis[di])
                        plt.draw()
                        plt.title('%d deg' % deg)

                        
                #plt.cla()
                #import bodyextractor
                #print('type:', type(vis[0]), 'dtype:', vis[0].dtype)
                #print('shape:', vis[0].shape)
                #bodyimg = bodyextractor.extract(vis[0][:,:,:3],tobgcolor=(0,0,0)) 
                #plt.imshow(bodyimg)
                plt.imshow(vis[0])
                plt.draw()
                plt.title('%s_%04d'%(dataset,ind))
                #_ = raw_input('next?')

                plt.savefig(out_path.replace('.pkl', '.png'))

	#break  # temp: one one image processing


if __name__ == '__main__':

    """  Parsing the arguments and load the SMPL specific model files    """ 

    logging.basicConfig(level=logging.INFO)

    # INPUT  & OPTIONS
    parser = argparse.ArgumentParser(description='run SMPLify on dataset')

    parser.add_argument(
        'base_dir',
        default='./',
        nargs='?',
        help="Directory that contains images/lsp and results/lps")
    parser.add_argument(
        '--dataset',
        default='lsp',
        type=str,
        help='dataset to process: default = lsp')
    parser.add_argument(
        '--out_dir',
        default='./smplifed/',
        type=str,
        help='Where results will be saved, default is /tmp/smplify_lsp')
    parser.add_argument(
        '--save',
        default=False,
        action='store_true',
        help="Save each step result.")    

    # Algorithm options    

    parser.add_argument(
        '--scale',
        default=1,
        type=int,
        help="if scaling up the input image, default = 1")
    parser.add_argument(
        '--no_interpenetration',
        default=False,
        action='store_true',
        help="Using this flag removes the interpenetration term, which speeds"
        "up optimization at the expense of possible interpenetration.")
    parser.add_argument(
        '--gender_neutral',
        default=False,
        action='store_true',
        help="Using this flag always uses the neutral SMPL model, otherwise "
        "gender specified SMPL models are used.")
    parser.add_argument(
        '--n_betas',
        default=10,
        type=int,
        help="Specify the number of shape coefficients to use.")
    parser.add_argument(
        '--flength',
        default=5000,
        type=float,
        help="Specify value of focal length.")
    parser.add_argument(
        '--side_view_thsh',
        default=25,
        type=float,
        help="This is thresholding value that determines whether the human is captured in a side view. If the pixel distance between the shoulders is less than this value, two initializations of SMPL fits are tried.")
    parser.add_argument(
        '--viz',
        default=False,
        action='store_true',
        help="Turns on visualization of intermediate optimization steps "
        "and final results.")
    args = parser.parse_args()


    #args.base_dir = join(join(args.base_dir, 'dataset'), '10k') # base dataset directory 
    args.base_dir = join(join(args.base_dir, 'dataset'), 'beforeafter') # base dataset directory 

    use_interpenetration = not args.no_interpenetration
    if not use_interpenetration:
        _LOGGER.info('Not using interpenetration term.')
    if args.gender_neutral:
        _LOGGER.info('Using gender neutral model.')

   
    # 1. load SMPL models (independent upon dataset)
    # Assumes 'models' in the 'code/' directory where this file is in.
    MODEL_DIR = join(abspath(dirname(__file__)), 'models')
    # Model paths:
    MODEL_NEUTRAL_PATH = join(MODEL_DIR, 'basicModel_neutral_lbs_10_207_0_v1.0.0.pkl')
    MODEL_FEMALE_PATH = join(MODEL_DIR, 'basicModel_f_lbs_10_207_0_v1.0.0.pkl')
    MODEL_MALE_PATH = join(MODEL_DIR, 'basicmodel_m_lbs_10_207_0_v1.0.0.pkl')

    if use_interpenetration:
        # paths to the npz files storing the regressors for capsules
        SPH_REGS_NEUTRAL_PATH = join(MODEL_DIR,
                                     'regressors_locked_normalized_hybrid.npz')
        SPH_REGS_FEMALE_PATH = join(MODEL_DIR,
                                    'regressors_locked_normalized_female.npz')
        SPH_REGS_MALE_PATH = join(MODEL_DIR,
                                  'regressors_locked_normalized_male.npz')

    # 2. Dataset dependent options 
    args.out_dir = join(args.base_dir, 'smplified')  #args.out_dir
 
    args.viz = True
    
    # 3. call the  main function
    print(">>> run options <<<")    
    print("    base_dir:%s, dataset:%s, out_dir:%s"%(args.base_dir, args.dataset, args.out_dir))
    print("    geneder:%r, out_dir:%s"%(args.gender_neutral, args.out_dir))
    print("    scale:%f,   save-step :%r"%(args.scale, args.save))
    
    #ch.ch.DEBUG = True ### To see the chumpy log 
    
    
    main(args.base_dir, args.out_dir, use_interpenetration, args.n_betas,
         args.flength, args.side_view_thsh, 'female', args.viz, args.dataset, args.scale, args.save)


