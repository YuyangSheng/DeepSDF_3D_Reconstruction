import os
import argparse
import numpy as np
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sdf_dataset import get_dataset
from autodecoder_sdf import AD_SDF


def loss_function(z, sdf_pre, sdf_gt, sigma):

    loss_l1 = torch.nn.L1Loss(reduction="sum")
    loss = loss_l1(sdf_pre, sdf_gt) + 1/sigma**2 * torch.sum(torch.square(z))

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a DeepSDF auto-decoder")
    parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
    parser.add_argument('--subsample', default=16384, type=int, help='number of points sampled from one model for training')
    parser.add_argument('--train_ratio', default=0.9, type=float, help='ratio of train set')
    parser.add_argument('--sigma', default=0.01, type=float, help='regularization parameter')
    parser.add_argument('--device', default="cpu", type=str, help='device used for training')
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs for training')
    parser.add_argument('--lr_net', default=1e-5, type=float, help='learning rate for training network')
    parser.add_argument('--lr_latent', default=1e-3, type=float, help='learning rate for training latent vectors')
    args = parser.parse_args()

    batch_size = args.batch_size
    subsample = args.subsample
    device = torch.device(args.device)
    print("using {} device.".format(device))
    epochs = args.epochs
    sigma = args.sigma
    lr_net = args.lr_net
    lr_latent = args.lr_latent

    rootpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "point_data")
    train_set, _ = get_dataset(rootpath, subsample, args.train_ratio, "train", True)
    train_loader = DataLoader(train_set,
                  batch_size=batch_size,
                  shuffle=True,
                  num_workers=0)
    models_num = train_set.__len__()
    print("using {} shapes for training.".format(models_num))

    z_dim = 256
    delta = 0.1
    latent_vectors = torch.nn.Embedding(models_num, z_dim).to(device)
    nn.init.normal_(latent_vectors.weight.data, mean=0.0, std=0.01)
    latent_vectors.requires_grad = True

    net = AD_SDF().to(device)
    optimizer = torch.optim.Adam([{"params": net.parameters(),
                    "lr": lr_net * batch_size},
                    
                    {"params": latent_vectors.parameters(),
                    "lr": lr_latent}])

    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    optimizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "optimizer")
    if not os.path.exists(optimizer_path):
        os.makedirs(optimizer_path)
    latent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latent")
    if not os.path.exists(latent_path):
        os.makedirs(latent_path)

    train_steps = len(train_loader)
    for epoch in range(epochs):        
        # training
        net.train()
        epoch_loss = 0.0
        for point_data, _, index in train_loader:
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
            loss = loss_function(batch_vectors, pred_sdf, sdf, sigma) / data_num

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss / train_steps

        print("train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, 
                                epochs,
                                epoch_loss))


        if (epoch + 1) % 10 == 0:
            torch.save({"epoch": epoch + 1, 
                        "model_state_dict": net.state_dict()},
                       os.path.join(model_path, f"decoder_{epoch + 1}.pth"))
            torch.save({"epoch": epoch + 1, 
                        "optimizer_state_dict": optimizer.state_dict()},
                       os.path.join(optimizer_path, f"optimizer_{epoch + 1}.pth"))
            torch.save({"epoch": epoch + 1, 
                        "latent_codes": latent_vectors},
                        os.path.join(latent_path, f"latent_{epoch + 1}.pth"))


    # save the final model
    torch.save({"epoch": epoch + 1, 
                "model_state_dict": net.state_dict()},
                os.path.join(model_path, f"decoder_{epoch + 1}.pth"))
    torch.save({"epoch": epoch + 1, 
                "optimizer_state_dict": optimizer.state_dict()},
                os.path.join(optimizer_path, f"optimizer_{epoch + 1}.pth"))
    torch.save({"epoch": epoch + 1, 
                "latent_codes": latent_vectors},
                os.path.join(latent_path, f"latent_{epoch + 1}.pth"))





