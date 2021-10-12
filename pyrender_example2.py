import trimesh
import pyrender

# from obj file to trimesh 
smpl_trimesh = trimesh.load('./models/smpl.obj')
print(dir(smpl_trimesh))

# trimesh to pyrender Mesh

mesh = pyrender.Mesh.from_trimesh(smpl_trimesh)
print(dir(mesh))


scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)

