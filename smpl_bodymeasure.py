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

def updateSMPL(scene, viewer, vertices, faces, faceColors = None):

    global smplMeshNode
    
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
    
    """  조인트 점찍는 방법 
    plot_joints = False  # TODO
    if plot_joints:
        sm = trimesh.creation.uv_sphere(radius=0.005)
        sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
        tfs = np.tile(np.eye(4), (len(joints), 1, 1))
        tfs[:, :3, 3] = joints
        joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        scene.add(joints_pcl)
    """
  
# 3 type of measure
# surface, segment, ??  
#  

body_landmarks = {
 "head":    [ (307, 138),    (138, 5428), (5428, 7028), (7028, 307) ],
 "waist":   [ (7803,1483),  (1483, 1111), (1111, 4971), (4971, 7803) ] }
'''
 "neck": [  ,  ,  ,],
 "shoulder": [        ],
 "bust": [        ],
 "hip": [   ],
 "
'''



def  measure_body(vertices, faces,  srcs):

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path

    # 1. make a graph using face and face distance 
    adjFaces = trimesh.graph.face_adjacency(faces, return_edges = False)
    
    nfaces = faces.shape[0]
    gmatrix = np.zeros((nfaces, nfaces), dtype = "float32")
    isFirst = True
    for i, (f1, f2) in enumerate(adjFaces):
    
        (v1, v2, v3) = faces[f1, :]   # vertices for f1 
        f1_center = (vertices[v1] + vertices[v2] + vertices[v3])/3.0  # face center 
        (v1, v2, v3) = faces[f2, :]   # vertices for f1 
        f2_center = (vertices[v1] + vertices[v2] + vertices[v3])/3.0  # face center 
        dist = np.linalg.norm(f1_center - f2_center)
        if isFirst:
            print("dist:", dist, " from ", f1_center, ",",  f2_center)
            isFirst = False
        gmatrix[f1, f2] = gmatrix[f1, f2]  = dist
        
    graph = csr_matrix(gmatrix)
   
    # 2. calclate the path 
    dist_matrix, predecessors = shortest_path(csgraph=graph, directed=False, indices=srcs, return_predecessors=True)
    
    return dist_matrix, predecessors
    
def buildPath(dist_list, pre_list, src, dst):   
    
    path_dist = dist_list[dst]   
   
    #print("path_dist:", path_dist)
    cur  = dst
    path = [dst]
    while True:    
        cur = pre_list[cur]
        path.append(cur)  
        if cur == src:
            break  
    
    return path, path_dist
 


if __name__ == "__main__":


    ## 1. Load SMPL model (here we load the female model)
    ## Make sure path is correct
    smpl_path = './models/basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    smpl = load_model(smpl_path)
    '''
    print("pose & beta size:", smpl.pose.size, ",", smpl.betas.size)
    print("pose :", smpl.pose)  # 24*3 = 72 (in 1-D)
    print("shape:", smpl.betas) # 10  (check the latent meaning..)
    '''
    ########################
    print("x-range (width): (%5.3f %5.3f) %5.3fm"%(np.min(smpl.r[:,0]), np.max(smpl.r[:,0]), np.max(smpl.r[:,0])- np.min(smpl.r[:,0]))) # x
    print("y-range (tall): (%5.3f %5.3f) %5.3fm"% (np.min(smpl.r[:,1]), np.max(smpl.r[:,1]), np.max(smpl.r[:,1])- np.min(smpl.r[:,1]))) # x
    print("z-range (thick): (%5.3f %5.3f) %5.3fm"%(np.min(smpl.r[:,2]), np.max(smpl.r[:,2]), np.max(smpl.r[:,2])- np.min(smpl.r[:,2]))) # x
    
    cam_pose = np.array([
        [1.0,  0.0,  0.0,  0.2],   # zero-rotation  
        [0.0,  1.0,  0.0,  0.0],   # 
        [0.0,  0.0,  1.0,  2.0],   # (0, 0, 2.0) translation  
        [0.0,  0.0,  0.0,  1.0]
    ])
    scene, viewer = setupScene(cam_pose)

    # body measure 
    srcs = [ ]
    for key in body_landmarks.keys():
        segments = body_landmarks[key]
        for i in range(len(segments)):
            srcs.append(segments[i][0])
    
    dist_matrix, predecessors = measure_body(smpl.r, smpl.f, srcs)


    colors = np.full(smpl.f.shape, fill_value = (127, 127, 127), dtype='uint8') # background 
    
    ccc = [(255,0,0), (0,0,255)] 
    srcIdx = 0
    
    
    for keyIdx, key in enumerate(body_landmarks.keys()):
        print(key)
        segments = body_landmarks[key]
        bodylen = 0.0
        for i in range(len(segments)):
            src = segments[i][0] 
            dst = segments[i][1] 
           
            #print(dist_matrix[srcIdx])
            #print(predecessors[srcIdx])
            
            path, dist = buildPath(dist_matrix[srcIdx], predecessors[srcIdx], src, dst)  
            print(src,  '=>', dst , ":",  bodylen)               
            bodylen = bodylen + dist
            srcIdx = srcIdx + 1
            for f in path:
                colors[f, : ] = ccc[keyIdx]
                
        print("total length:",  bodylen)   
        
    # d
    updateSMPL(scene, viewer, smpl.r, smpl.f, colors) 
    
    x = input('finish')
         
    # Close the viwer (and its thread) to finish program     
    viewer.close_external()
    while viewer.is_active:
        pass
               
  
        