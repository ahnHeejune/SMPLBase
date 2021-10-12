###################################################################
# PyReder Pyramid demo 
###################################################################
# 3D Triangle Mesh model 
# 3D mesh model files
# 
# (c) 2020  heejune@seoultech.ac.kr
###################################################################

print("Pyredner Pyramid Rendering Demo") 


import math
import numpy as np
'''
from pyrender import PerspectiveCamera,\
                     DirectionalLight, SpotLight, PointLight,\
                     MetallicRoughnessMaterial,\
                     Primitive, Mesh, Node, Scene,\
                     Viewer, OffscreenRenderer, RenderFlags
'''
import pyrender
import trimesh

def createScene(cam_pose):

    # setup scene and camera  
    scene = pyrender.Scene( bg_color=np.array([0.0, 0.0, 0.0]),  
                            ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
    cam = pyrender.PerspectiveCamera(yfov=(np.pi* 75.0/ 180.0))     # 90 degree FOV
    cam_node = scene.add(cam, pose = cam_pose)
    
    return scene    
    
def createMesh( vertices, faces,   # Geometry 
                texture = None, f_colors = None, v_colors = None ):
    # 1. color setting     
    if texture is not None:
        raise Exception('not yet support texture (TODO)')
    elif f_colors is not None:
        tri_mesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors)
    elif v_colors is not None:
        v_colors = np.ones([vertices.shape[0], 4]) * [0.7, 0.7, 0.7, 0.8]   
        tri_mesh = trimesh.Trimesh(vertices, faces, vertex_colors=v_colors)
    else:
        f_colors = np.ones([faces.shape[0], 4]) * [0.9, 0.9, 0.9, 1.0]                  
        tri_mesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors)
        
    # 2. build mesh 
    mesh = pyrender.Mesh.from_trimesh(tri_mesh, smooth = False)    
    
    return mesh
 
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
   

    

def  testDynamicMesh(onScreen = True):

    """ simple 3-D mesh  visualization """

    v1 = np.array([[+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 3.0, 0.0]]) # 4 vertices
    v2 = np.array([[+2.0, 0.0, 0.0], [-2.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 2.0, 0.0]]) # 4 vertices
 
    f = np.array([[0,1,2], [1,0, 3], [1, 3, 2], [0, 2, 3]])          # 4 faces (Face Direction is crucial)
     
    '''
    colors = np.ones([v.shape[0], 4]) * [0.7, 0.7, 0.7, 0.8] 
    #print(np.ones([v.shape[0], 4]))
    #print(colors)
    '''
    f_colors = np.ones([f.shape[0], 4]) * [0.0, 0.0, 0.0, 1.0]   
    f_colors[0, 0] = 1.0  # red bottom
    f_colors[1, 1] = 1.0  # green front 
    f_colors[2, 2] = 1.0  # blue  left
    f_colors[3, :] = 1.0  # white right 
    print("fcolors:", f_colors)
   
    
    numViewAngles = 1
   
    for i in range(numViewAngles):  
        
        z_angle = np.pi*i/8
        c = math.cos(z_angle)
        s = math.sin(z_angle)
        r = 4.0
        x = r*s
        z = r*c
        cam_pose = np.array([
            [c,   0,  s,     x],    # rotation around z axis 
            [0,   1,  0.0,  0.0],   # 
            [-s,   0,  c,  z],      # (x, 0, z) translation  
            [0.0,  0.0,  0.0,  1.0]
        ])
        
    
   
    scene = createScene(cam_pose)
               
    mesh = createMesh(v1, f, f_colors = f_colors)           
    mesh_node = scene.add(mesh)
    
    if onScreen:
       
        import time
        
        view = pyrender.Viewer(scene, use_raymond_lighting=True, run_in_thread = True)
        
        for i in range(40):
            
            pose = np.array([
                [1,   0,  0,     i*0.2],    #  
                [0,   1,  0.0,  0.0],   # 
                [0,   0,  1,    0],      # (x, 0, z) translation  
                [0.0,  0.0,  0.0,  1.0]
            ])
            view.render_lock.acquire()
            #scene.set_pose(mesh_node, pose)
            if i%2 == 0:
                scene.remove_node(mesh_node)
                mesh = createMesh(v2, f, f_colors = f_colors)           
                mesh_node = scene.add(mesh)
            else:
                scene.remove_node(mesh_node)
                mesh = createMesh(v1, f, f_colors = f_colors)           
                mesh_node = scene.add(mesh)
            
            view.render_lock.release()
            time.sleep(1)
            
        # close the viewer    
        view.close_external()
        while view.is_active:
            pass
            
   
        
        
def  testStaticMesh(onScreen = True):

    """ simple 3-D mesh  visualization """

    v1 = np.array([[+1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 3.0, 0.0]]) # 4 vertices
    f = np.array([[0,1,2], [1,0, 3], [1, 3, 2], [0, 2, 3]])          # 4 faces (Face Direction is crucial)
    
    '''
    colors = np.ones([v.shape[0], 4]) * [0.7, 0.7, 0.7, 0.8] 
    #print(np.ones([v.shape[0], 4]))
    #print(colors)
    '''
    f_colors = np.ones([f.shape[0], 4]) * [0.0, 0.0, 0.0, 1.0]   
    f_colors[0, 0] = 1.0  # red bottom
    f_colors[1, 1] = 1.0  # green front 
    f_colors[2, 2] = 1.0  # blue  left
    f_colors[3, :] = 1.0  # white right 
    print("fcolors:", f_colors)
   
    
    numViewAngles = 1
   
    for i in range(numViewAngles):  
        
        z_angle = np.pi*i/8
        c = math.cos(z_angle)
        s = math.sin(z_angle)
        r = 4.0
        x = r*s
        z = r*c
        cam_pose = np.array([
            [c,   0,  s,     x],    # rotation around z axis 
            [0,   1,  0.0,  0.0],   # 
            [-s,   0,  c,  z],      # (x, 0, z) translation  
            [0.0,  0.0,  0.0,  1.0]
        ])
        
    
   
    scene = createScene(cam_pose)
               
    mesh = createMesh(v1, f, f_colors = f_colors)           
    mesh_node = scene.add(mesh)
   
    if onScreen:
    
        view = pyrender.Viewer(scene, use_raymond_lighting=True)
        
    else:  # offScreen Rendering doesnot work on Windows 
    
        r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480, point_size=1.0)
        color, depth = r.render(scene)
        import matplotlib.pyplot as plt
        plt.subplot(1,2,1)
        plt.imsow(color)
        plt.subplot(1,2,2)
        plt.imsow(depth)
        plt.ishow()
            
        

if __name__ == "__main__":

    testStaticMesh(True)
    
    #testDynamicMesh(True)
    
    
