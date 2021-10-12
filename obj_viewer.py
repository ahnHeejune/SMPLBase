"""

## not important ###
"Unable to load numpy_formathandler accelerator from OpenGL_accelerate"
I have solved this issue by removing the accelerate library since it's the cause of the bug pip uninstall PyOpenGL_accelerate – Bilal Sep 25 '20 at 7:07
"""
import os
import numpy as np
import trimesh
import sys 

from pyrender import  Mesh, Node, Scene, Viewer

def showObjFile(objFilePath):

    trimesh1 = trimesh.load(objFilePath)
    mesh = Mesh.from_trimesh(trimesh1)

    # Scene creation
    scene = Scene(ambient_light=np.array([1., 1.0, 1.0, 1.0])) ##### I increased this 

    node = Node(mesh=mesh, translation=np.array([0.1, 0.15, -np.min(trimesh1.vertices[:,2])]))
    scene.add_node(node)

    v = Viewer(scene, shadows=True)


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("usage:  %s  objfile"%sys.argv[0])
        exit()    
    if not os.path.exists(sys.argv[1]):
        print("Cannot find the file ")
        exit()
        
    showObjFile(sys.argv[1])
    