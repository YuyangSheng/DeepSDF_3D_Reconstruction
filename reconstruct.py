import torch
import numpy as np
import argparse
import time
import os
from sdf_dataset import get_dataset
import logging
import torch.nn as nn
from torch.utils.data import DataLoader
from autodecoder_sdf import AD_SDF
import skimage.measure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# from tools import loss_function
import plyfile


def loss_function(z, sdf_pre, sdf_gt, sigma):

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss = loss_l1(sdf_pre, sdf_gt) + 1/sigma**2 * torch.sum(torch.square(z))

    return loss


def create_mesh(net, latent_vecs, test_indices, save_dir, resolution, device):

    # NOTE: the voxel_origin is actually the (bottom, left, down) corner, not the middle
    voxel_origin = [-1, -1, -1]
    voxel_size = 2.0 / (resolution - 1)

    # sample voxles in a unit cube, bottom, left, down is -1 and top, right, up is 1
    overall_index = torch.arange(0, resolution ** 3, 1, out=torch.LongTensor())
    samples = torch.zeros(resolution ** 3, 4)
    samples[:, 2] = overall_index % resolution
    samples[:, 1] = (overall_index.long() / resolution) % resolution
    samples[:, 0] = ((overall_index.long() / resolution) / resolution) % resolution

    samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
    samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
    samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]

    samples.requires_grad = False

    num_samples = resolution ** 3
    splits = 10
    splits_size = int(num_samples / splits)
    net.eval()
    latent_vecs.requires_grad = False
    models_num = len(test_indices)

    for ids in range(models_num):
        print(ids)
        point_samples = torch.clone(samples.detach())
        latent_inputs = latent_vecs(torch.tensor(ids).to(device)).expand(num_samples, -1).chunk(splits)
        points_inputs = point_samples[:, :3].to(device).chunk(splits)
        for i in range(splits):
            recon_inputs = torch.cat([latent_inputs[i], points_inputs[i]], 1).to(torch.float)
            pred_sdf = net(recon_inputs)
            pred_sdf = torch.abs(torch.clamp(pred_sdf, -delta, delta)).squeeze()
            point_samples[i*splits_size : (i+1)*splits_size, 3] = pred_sdf
        
        sdf = point_samples[:, 3]
        sdf = sdf.reshape(resolution, resolution, resolution).data.cpu().numpy()

        vertices, faces, normals, values = skimage.measure.marching_cubes(
            sdf, level=None, spacing=[voxel_size] * 3)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlim(np.min(vertices[:,0]), np.max(vertices[:,0]))
        ax.set_ylim(np.min(vertices[:,1]), np.max(vertices[:,1])) 
        ax.set_zlim(np.min(vertices[:,2]), np.max(vertices[:,2]))

        mesh = Poly3DCollection(vertices[faces])
        mesh.set_edgecolor('k')
        ax.add_collection3d(mesh)
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join(save_dir, f"mesh_{test_indices[ids]}.png"))

        # write mesh into .ply file
        mesh_points = np.zeros_like(vertices)
        mesh_points[:, 0] = voxel_origin[0] + vertices[:, 0]
        mesh_points[:, 1] = voxel_origin[1] + vertices[:, 1]
        mesh_points[:, 2] = voxel_origin[2] + vertices[:, 2]

        num_verts = vertices.shape[0]
        num_faces = faces.shape[0]

        verts_tuple = np.zeros((num_verts,), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])

        for i in range(0, num_verts):
            verts_tuple[i] = tuple(mesh_points[i, :])

        faces_building = []
        for i in range(0, num_faces):
            faces_building.append(((faces[i, :].tolist(),)))
        faces_tuple = np.array(faces_building, dtype=[("vertex_indices", "i4", (3,))])

        el_verts = plyfile.PlyElement.describe(verts_tuple, "vertex")
        el_faces = plyfile.PlyElement.describe(faces_tuple, "face")

        ply_data = plyfile.PlyData([el_verts, el_faces])
        ply_data.write(os.path.join(save_dir, f"mesh_{test_indices[ids]}.ply"))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Reconstruction a model using the trained DeepSDF auto-decoder")
    parser.add_argument('--mod', default="test", type=str, help='train or test, using train or test set for reconstruction')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for testing')
    parser.add_argument('--subsample', default=20000, type=int, help='number of points sampled from one model for training')
    parser.add_argument('--sigma', default=0.01, type=float, help='regularization parameter')
    parser.add_argument('--train_ratio', default=0.9, type=float, help='ratio of train set')
    parser.add_argument('--device', default="cuda", type=str, help='device used for testing')
    parser.add_argument('--weights', default="decoder_60.pth", type=str, help='the filename of auto-decoder paremeters')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs for testing')
    parser.add_argument('--resolution', default=50, type=int, help='resolution of reconstruction')
    parser.add_argument('--lr_latent', default=1e-3, type=float, help='learning rate for testing latent vectors')
    parser.add_argument('--latent_size', default=256, type=int, help='the dimension of latent vector for reconstruction')
    args = parser.parse_args()

    device = torch.device(args.device)
    print("using {} device.".format(device))
    subsample = args.subsample
    epochs = args.epochs
    batch_size = args.batch_size

    mesh_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"mesh_{args.mod}")
    if not os.path.exists(mesh_dir):
        os.makedirs(mesh_dir)

    rootpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "point_data")
    test_set, test_indices = get_dataset(rootpath, subsample, args.train_ratio, args.mod, True)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=0)

    models_num = len(test_set)
    test_steps = len(test_loader)
    print("using {} shapes for reconstruction.".format(models_num))
    
    delta = 0.1
    latent_vectors = torch.nn.Embedding(models_num, args.latent_size).to(device)
    nn.init.normal_(latent_vectors.weight.data, mean=0.0, std=0.01)
    latent_vectors.requires_grad = True

    # Load model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights", args.weights)
    if args.device == "cpu":
        model_state = torch.load(model_path, map_location=torch.device('cpu'))
    else:
        model_state = torch.load(model_path)

    net = AD_SDF().to(device)
    net.load_state_dict(model_state["model_state_dict"])
    for param in net.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam([{"params": latent_vectors.parameters(),
                                   "lr": args.lr_latent}])
    
    for test_data, _, test_index in test_loader:
        net.eval()
        model_loss = 0.0
        for epoch in range(epochs):
            # backward with partial model
            index = torch.clone(test_index)
            point_data = torch.clone(test_data)
            data_num = point_data.shape[1]
            point_data = point_data.reshape(-1, 4).to(device)
            point_data.requires_grad = False
            batch_vectors = latent_vectors(index.to(device)).to(device)

            xyz = point_data[:, :3]
            sdf = torch.clamp(point_data[:, 3], -delta, delta).unsqueeze(1)
            index = index.unsqueeze(-1).repeat(1, data_num).view(-1).to(device)
            input_vectors = latent_vectors(index).to(device)

            inputs = torch.cat([input_vectors, xyz], dim=1).to(torch.float)
            pred_sdf = net(inputs)
            pred_sdf = torch.clamp(pred_sdf, -delta, delta)
            loss = loss_function(batch_vectors, pred_sdf, sdf, args.sigma) / data_num

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            model_loss += loss / test_steps
        
        logging.debug(f"Epoch {epoch}: Loss = {model_loss}")

    create_mesh(net, latent_vectors, test_indices, mesh_dir, resolution=args.resolution, device=device)



    
