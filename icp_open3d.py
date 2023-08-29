import copy

import numpy as np
import open3d as o3d
import torch
import time
from stl import mesh
import stl

def save2stl(vertices,faces,path):
    shape = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            shape.vectors[i][j] = vertices[f[j], :]
    shape.save(path,mode=stl.Mode.ASCII)

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw([source_temp, target_temp])

def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def main():
    voxel_size = 0.005
    print(":: Load two mesh.")
    target_mesh = o3d.io.read_triangle_mesh('./exp3/normalized_left.stl')
    source_mesh = o3d.io.read_triangle_mesh('./exp3/normalized_random.stl')

    draw_registration_result(target_mesh, source_mesh, np.identity(4))

    print(":: Sample mesh to point cloud")
    target = target_mesh.sample_points_uniformly(5000)
    source = source_mesh.sample_points_uniformly(5000)

    target = o3d.t.geometry.PointCloud.from_legacy(target)
    source = o3d.t.geometry.PointCloud.from_legacy(source)
    draw_registration_result(source, target, np.identity(4))

    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = 2

    # Initial alignment or source to target transform.
    init_source_to_target = np.asarray([[0.862, 0.011, -0.507, 0.5],
                                        [-0.139, 0.967, -0.215, 0.7],
                                        [0.487, 0.255, 0.835, -1.4],
                                        [0.0, 0.0, 0.0, 1.0]])


    treg = o3d.t.pipelines.registration


    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = treg.TransformationEstimationPointToPlane()

    # Convergence-Criteria for Vanilla ICP
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001,
                                        relative_rmse=0.000001,
                                        max_iteration=500)

    criteria_list = [
        treg.ICPConvergenceCriteria(relative_fitness=0.0001,
                                    relative_rmse=0.0001,
                                    max_iteration=20),
        treg.ICPConvergenceCriteria(0.00001, 0.00001, 15),
        treg.ICPConvergenceCriteria(0.000001, 0.000001, 10)
    ]

    # Down-sampling voxel-size.
    voxel_size = 0.025
    voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])


    # Save iteration wise `fitness`, `inlier_rmse`, etc. to analyse and tune result.
    save_loss_log = True

    s = time.time()

    callback_after_iteration = lambda updated_result_dict : print("Iteration Index: {}, Fitness: {}, Inlier RMSE: {},".format(
    updated_result_dict["iteration_index"].item(),
    updated_result_dict["fitness"].item(),
    updated_result_dict["inlier_rmse"].item()))

    registration_icp = treg.icp(source, target, max_correspondence_distance,
                    init_source_to_target, estimation, criteria,
                    voxel_size, callback_after_iteration)

    icp_time = time.time() - s
    print("Time taken by ICP: ", icp_time)
    print("Inlier Fitness: ", registration_icp.fitness)
    print("Inlier RMSE: ", registration_icp.inlier_rmse)

    draw_registration_result(source, target, registration_icp.transformation)

    RTs = registration_icp.transformation.numpy()

    np.save('./exp3/RTs_open3d.npy',RTs)

    # 齐次坐标
    source_mesh.remove_duplicated_vertices()
    point = np.asarray(source_mesh.vertices)

    source_vert = torch.cat([torch.tensor(point),torch.ones(point.shape[0],1)],dim=-1)
    res_vert = source_vert @ torch.transpose(torch.tensor(RTs),0,1)

    save2stl(res_vert[:,:3].detach().cpu().numpy(),np.asarray(source_mesh.triangles),f'./exp3/icp-open3d-left.stl')

    
main()
