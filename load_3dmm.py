import os
from scipy.io import loadmat
import torch
import numpy as np
from stl import mesh
import stl
import transformation


def save2stl(vertices,faces,path):
    shape = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            shape.vectors[i][j] = vertices[f[j], :]
    shape.save(path,mode=stl.Mode.ASCII)

ear3dmm = loadmat('earmodel.mat')

mean_shape = ear3dmm['mu5']    # 1x21333
basis = ear3dmm['coeff5']      # 21333x499
latent = ear3dmm['latent5']    # 499x1

triangle = ear3dmm['sourceF1'] # 14026x3
triangle -= 1


# # 随机生成10个耳朵3d shape
# os.makedirs(f'./exp1',exist_ok=True)
# for i in range(10):
#     tmp_code = np.random.randn(499,1)

#     tmp_offset = torch.tensor(basis).to('cuda') @ torch.tensor(tmp_code).to('cuda') # 21333x1

#     tmp_offset *= 2

#     tmp_vert = torch.tensor(mean_shape).to('cuda') + tmp_offset.view(1,21333) # 21333x1

#     tmp_vert = tmp_vert.view(7111,3)

#     save2stl(tmp_vert.detach().cpu().numpy(),triangle,f'./exp1/{i}.stl')

# rigid transformation
# 随机生成一个shape
os.makedirs(f'./exp2',exist_ok=True)
tmp_code = np.random.randn(499,1)
tmp_offset = torch.tensor(basis).to('cuda') @ torch.tensor(tmp_code).to('cuda') # 21333x1
tmp_vert = torch.tensor(mean_shape).to('cuda') + tmp_offset.view(1,21333) # 21333x1
tmp_vert = tmp_vert.view(7111,3).float()
save2stl(tmp_vert.detach().cpu().numpy(),triangle,f'./exp2/origin.stl')

scale = 2
translation = [0,10,0]
angles_xyz = [0,0,90]

angles_tensor = torch.tensor(np.deg2rad(angles_xyz)).float().view(1,3).to('cuda')
translation_tensor = torch.tensor(translation).float().view(1,3).to('cuda')
scale_tensor = torch.tensor(scale).view(1,1).float().to('cuda')

rotate_vert = transformation.transformation(tmp_vert.view(1,-1,3),angles=angles_tensor)
save2stl(rotate_vert[0].detach().cpu().numpy(),triangle,f'./exp2/rotate.stl')

scale_vert = transformation.transformation(tmp_vert.view(1,-1,3),scale=scale_tensor)
save2stl(scale_vert[0].detach().cpu().numpy(),triangle,f'./exp2/scale.stl')

translate_vert = transformation.transformation(tmp_vert.view(1,-1,3),translation=translation_tensor)
save2stl(translate_vert[0].detach().cpu().numpy(),triangle,f'./exp2/translate.stl')


