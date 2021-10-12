import trimesh
import pyrender

# from obj file to trimesh 
fuze_trimesh = trimesh.load('./models/fuze.obj')
print(dir(fuze_trimesh))

# trimesh to pyrender Mesh

mesh = pyrender.Mesh.from_trimesh(fuze_trimesh)
print(dir(mesh))


scene = pyrender.Scene()
scene.add(mesh)
pyrender.Viewer(scene, use_raymond_lighting=True)

