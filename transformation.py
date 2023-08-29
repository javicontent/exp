import torch

def compute_rotation(angles):
    """
    Return:
        rot              -- torch.tensor, size (B, 3, 3) pts @ trans_mat

    Parameters:
        angles           -- torch.tensor, size (B, 3), radian
    """
    device = angles.device
    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(device)
    zeros = torch.zeros([batch_size, 1]).to(device)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],
    
    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x), 
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])
    
    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)

def transformation(vertices,
                    angles=torch.tensor([0,0,0]).view(1,3),
                    scale=torch.tensor([1]).view(1,1),
                    translation=torch.tensor([0,0,0]).view(1,3)):

    angles = angles.to(vertices.device)
    scale = scale.to(vertices.device)
    translation = translation.to(vertices.device)

    ## scale
    center = torch.mean(vertices,1) # B, Nv, 3
    vertices_new = vertices - center
    vertices_new = vertices_new * scale
    vertices_new = vertices_new + center

    ## rotate
    R = compute_rotation(angles)
    vertices_new = vertices_new @ R

    ## translate
    vertices_new = vertices_new + translation.view(1,1,3)

    return vertices_new

def _apply_similarity_transform(
    X: torch.Tensor, R: torch.Tensor, T: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """
    Applies a similarity transformation parametrized with a batch of orthonormal
    matrices `R` of shape `(minibatch, d, d)`, a batch of translations `T`
    of shape `(minibatch, d)` and a batch of scaling factors `s`
    of shape `(minibatch,)` to a given `d`-dimensional cloud `X`
    of shape `(minibatch, num_points, d)`
    """
    X = s[:, None, None] * torch.bmm(X, R) + T[:, None, :]
    return X