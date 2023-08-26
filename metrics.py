import os
import numpy as np
import argparse
import open3d as o3d
from scipy.spatial import cKDTree as KDTree
from scipy.stats import wasserstein_distance


def chamfer_distance(gt_pcd, recon_pcd, num_samples=30000):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_mesh: trimesh.points.PointCloud of just poins, sampled from the surface (see
             compute_metrics.ply for more documentation)

    recon_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
                method (see compute_metrics.py for more)

    """

    gt_samples = np.asarray(gt_pcd.points)
    indices = np.arange(gt_samples.shape[0])
    np.random.shuffle(indices)
    gt_samples = gt_samples[indices][:num_samples, :]

    recon_samples = np.asarray(recon_pcd.points)
    indices = np.arange(recon_samples.shape[0])
    np.random.shuffle(indices)
    recon_samples = recon_samples[indices][:num_samples, :]

    # one direction
    recon_kdtree = KDTree(recon_samples)
    distances_1, _ = recon_kdtree.query(gt_samples)
    gt_to_recon_chamfer = np.mean(np.square(distances_1))

    # other direction
    gt_kdtree = KDTree(gt_samples)
    distances_2, _ = gt_kdtree.query(recon_samples)
    recon_to_gt_chamfer = np.mean(np.square(distances_2))

    return gt_to_recon_chamfer + recon_to_gt_chamfer


def earth_mover_distance(gt_pcd, recon_pcd, num_samples=500):
    """
    This function computes a symmetric chamfer distance, i.e. the sum of both chamfers.

    gt_mesh: trimesh.points.PointCloud of just poins, sampled from the surface (see
             compute_metrics.ply for more documentation)

    recon_mesh: trimesh.base.Trimesh of output mesh from whichever autoencoding reconstruction
                method (see compute_metrics.py for more)

    """

    threshold = 0.02
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0], 
                             [0.0, 0.0, 0.0, 1.0]])
    
    evaluation = o3d.pipelines.registration.evaluate_registration(recon_pcd, gt_pcd, 
                                                        threshold, trans_init)

    gt_recon_icp = o3d.pipelines.registration.registration_icp(
        recon_pcd, gt_pcd, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    recon_pcd_trans = o3d.geometry.PointCloud()
    recon_pcd_trans.points = recon_pcd.points
    recon_pcd_trans.colors = recon_pcd.colors
    recon_pcd_trans.normals = recon_pcd.normals
    recon_pcd_trans.transform(gt_recon_icp.transformation)

    gt_samples = np.asarray(gt_pcd.points)
    recon_samples = np.asarray(recon_pcd.points)
    recon_trans_samples = np.asarray(recon_pcd_trans.points)

    # other direction
    gt_kdtree = KDTree(gt_samples)
    distances, _ = gt_kdtree.query(recon_trans_samples)
    dis_indices = np.argsort(distances)
    points_distances = recon_trans_samples[dis_indices][:num_samples] - recon_samples[dis_indices][:num_samples]
    emd_distances = np.mean(np.square(points_distances))
    
    return emd_distances


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute CD and EMD")
    parser.add_argument('--recon_folder', default="mesh_test", type=str, help='folder name containing reconstructed meshes')
    parser.add_argument('--gt_folder', default="mesh", type=str, help='folder name containing ground truth meshes')
    args = parser.parse_args()

    recon_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", args.recon_folder)
    gt_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", args.gt_folder)

    recon_files = sorted(os.listdir(recon_root))
    gt_files = sorted(os.listdir(gt_root))
    file_num = 0

    cd_ave = 0.0
    emd_ave = 0.0
    for file_name in recon_files:
        if file_name.endswith('.ply'):
            indices = int(file_name[5:-4])
            recon_file = os.path.join(recon_root, file_name)
            gt_file = os.path.join(gt_root, gt_files[indices])

            recon_pcd = o3d.io.read_point_cloud(recon_file)
            gt_mesh = o3d.io.read_triangle_mesh(gt_file)
            gt_pcd = o3d.geometry.PointCloud()
            gt_pcd.points = gt_mesh.vertices
            gt_pcd.colors = gt_mesh.vertex_colors
            gt_pcd.normals = gt_mesh.vertex_normals

            file_num += 1
            cd_ave += chamfer_distance(gt_pcd, recon_pcd)
            emd_ave += earth_mover_distance(gt_pcd, recon_pcd)

    cd_ave = cd_ave / file_num
    emd_ave = emd_ave / file_num
    print("CD:", cd_ave)
    print("EMD:", emd_ave)

