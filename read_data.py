import os, sys
from tqdm import tqdm
import trimesh
import numpy as np
import glob
from datasets import SampleFromMesh


if __name__ == '__main__':
    
    DATA_DIR = os.path.join(os.getcwd(), 'data', 'ShapeNetCore.v2')
    SAVE_DIR = os.path.join(os.getcwd(), 'data', 'data')
    DISTANT_SAVE_DIR = os.path.join(os.getcwd(), 'data', 'distant_samples')

    points_dir = os.path.join(SAVE_DIR, 'points')
    noisy_points_dir = os.path.join(SAVE_DIR, 'noisy_points')
    sdf_dir = os.path.join(SAVE_DIR, 'sdf')
    mesh_dir = os.path.join(SAVE_DIR, 'mesh')

    dist_pts_dir = os.path.join(DISTANT_SAVE_DIR, 'points')
    dist_noisy_pts_dir = os.path.join(DISTANT_SAVE_DIR, 'noisy_points')
    dist_sdf_dir = os.path.join(DISTANT_SAVE_DIR, 'sdf')
    

    # make directories for processed data
    os.makedirs(points_dir, exist_ok=True)
    os.makedirs(noisy_points_dir, exist_ok=True)
    os.makedirs(sdf_dir, exist_ok=True)
    os.makedirs(mesh_dir, exist_ok=True)

    os.makedirs(dist_pts_dir, exist_ok=True)
    os.makedirs(dist_noisy_pts_dir, exist_ok=True)
    os.makedirs(dist_sdf_dir, exist_ok=True)
    
    idx = 0
    # read sample points on the surface of the mesh
    for category in os.listdir(DATA_DIR)[:1]:
        # save dir path of each category
        category_dir = os.path.join(DATA_DIR, category)

        for source_dir in tqdm(glob.glob(category_dir+'/*')[idx:1000]):
            # read all .obj file
            mesh_file_name = list(glob.iglob(source_dir + "/**/*.obj"))+ list(glob.iglob(source_dir + "/*.obj"))
                                                                            
            mesh = trimesh.load(mesh_file_name[0], force='mesh')
            sample_mesh = SampleFromMesh(mesh, num_pts=20000)

            # save data
            np.save(os.path.join(points_dir, str(idx)+'.npy'), sample_mesh.sample_points)
            np.save(os.path.join(noisy_points_dir, str(idx)+'.npy'), sample_mesh.noisy_points)
            np.save(os.path.join(sdf_dir, str(idx)+'.npy'), sample_mesh.sdf)
            # shutil.copy(mesh_file_name[0], os.path.join(mesh_dir, str(idx)+'.obj'))
            idx += 1
    

    # Load distant samples from space
    idx = 0
    for obj_name in os.listdir(mesh_dir)[idx:]:
        mesh_file_name = os.path.join(mesh_dir, obj_name)
        mesh = trimesh.load(mesh_file_name, force='mesh')
        dist_sample_mesh = SampleFromMesh(mesh, is_distant=True, num_pts=5000)
        # print(dist_sample_mesh.sdf.shape, dist_sample_mesh.sdf.max(), dist_sample_mesh.sdf.min())

        # save data
        idx_str = obj_name.split('.')[0].strip()
        np.save(os.path.join(dist_pts_dir, idx_str+'.npy'), dist_sample_mesh.sample_points)
        np.save(os.path.join(dist_noisy_pts_dir, idx_str+'.npy'), dist_sample_mesh.noisy_points)
        np.save(os.path.join(dist_sdf_dir, idx_str+'.npy'), dist_sample_mesh.sdf)
        print(idx)
        idx += 1  


    # Merge two datasets
    # create path for saving merged data
    merged_save_path = os.path.join(os.getcwd(), 'data', 'merged_data')
    merged_noisy_pts_path = os.path.join(merged_save_path, 'noisy_points')
    merged_sdf_path = os.path.join(merged_save_path, 'sdf')

    if not os.path.exists(merged_noisy_pts_path):
        os.makedirs(merged_noisy_pts_path)
    if not os.path.exists(merged_sdf_path):
        os.makedirs(merged_sdf_path)

    # read data from two directories and merge data with same file names
    for filename in os.listdir(dist_noisy_pts_dir):
        pts_path = glob.glob(noisy_points_dir+'/'+filename)[0]
        dist_pts_path = os.path.join(dist_noisy_pts_dir, filename)

        sdf_path = glob.glob(sdf_dir+'/'+filename)[0]
        dist_sdf_path = os.path.join(dist_sdf_dir, filename)
        
        points = np.load(pts_path)
        dist_points = np.load(dist_pts_path)
        merged_points = np.vstack([points, dist_points])

        sdf = np.load(sdf_path)
        dist_sdf = np.load(dist_sdf_path)
        merged_sdf = np.vstack([sdf[:, np.newaxis], dist_sdf[:, np.newaxis]])

        np.save(os.path.join(merged_noisy_pts_path, filename), merged_points)
        np.save(os.path.join(merged_sdf_path, filename), merged_sdf.squeeze())