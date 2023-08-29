import pytorch3d.ops
from stl import mesh
from scipy.io import loadmat
import numpy as np
import torch
import trimesh
import transformation
import stl
def save2stl(vertices,faces,path):
    shape = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            shape.vectors[i][j] = vertices[f[j], :]
    shape.save(path,mode=stl.Mode.ASCII)


# load 3dmm
ear3dmm = loadmat('earmodel.mat')
mean_shape = ear3dmm['mu5']    # 1x21333
basis = ear3dmm['coeff5']      # 21333x499
latent = ear3dmm['latent5']    # 499x1
triangle = ear3dmm['sourceF1'] # 14026x3
triangle -= 1

# a rand ear
tmp_code = np.random.randn(499,1)
tmp_offset = torch.tensor(basis).to('cuda') @ torch.tensor(tmp_code).to('cuda') # 21333x1
tmp_vert = torch.tensor(mean_shape).to('cuda') + tmp_offset.view(1,21333) # 21333x1
tmp_vert = tmp_vert.view(1,7111,3).float()

save2stl(tmp_vert[0].detach().cpu().numpy(),triangle,f'./exp3/random.stl')

# normalize the scale of rand ear
rand_ear = trimesh.Trimesh(vertices=tmp_vert[0].detach().cpu().numpy(),
                       faces=triangle)

tmp_vert -= torch.tensor(rand_ear.centroid).view(1,1,3).to('cuda').float()
tmp_vert /= rand_ear.scale

save2stl(tmp_vert[0].detach().cpu().numpy(),triangle,f'./exp3/normalized_random.stl')


# load the ct right ear and normlize it
ct_ear = trimesh.load_mesh('./exp3/left.stl')
left = True

# normlize
normlized_vertices = ct_ear.vertices - ct_ear.centroid
normlized_vertices = normlized_vertices / ct_ear.scale

if left:
    normlized_vertices[:,1] = 1-normlized_vertices[:,1]
    normlized_vertices = normlized_vertices - np.mean(normlized_vertices,0)

target_vert = torch.tensor(normlized_vertices).to('cuda').view(1,-1,3).float()


# icp
icp_res = pytorch3d.ops.iterative_closest_point(tmp_vert,target_vert,max_iterations=1000,estimate_scale=False,allow_reflection=False,verbose=True)

# perform RTS
res_vert = transformation._apply_similarity_transform(tmp_vert.view(1,-1,3),R=icp_res.RTs.R,T=icp_res.RTs.T,s=icp_res.RTs.s)

# save
save2stl(target_vert[0].detach().cpu().numpy(),ct_ear.faces,f'./exp3/normalized_left.stl')

save2stl(res_vert[0].detach().cpu().numpy(),triangle,f'./exp3/icp.stl')
