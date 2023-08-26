
import numpy as np
import trimesh

class SampleFromMesh():
    def __init__(self, mesh, is_distant=False, num_pts=500000):
        self.mesh = mesh
        if(is_distant == False):
            self.sample_points = self.sampling(num_pts)
        else:
            self.sample_points = self.sampling_distant_points(num_pts)

        self.get_sdf()

    def sampling(self, num_pts=500000):
        sample_points, face_indices = trimesh.sample.sample_surface(self.mesh, count=num_pts)
        return sample_points
    
    def sampling_distant_points(self, num_pts):
        bbx = self.mesh.bounds * 1.2

        x = np.random.uniform(bbx[0, 0], bbx[1, 0], size=(num_pts, 1))
        y = np.random.uniform(bbx[0, 1], bbx[1, 1], size=(num_pts, 1))
        z = np.random.uniform(bbx[0, 2], bbx[1, 2], size=(num_pts, 1))
        sample_points = np.hstack([x, y, z])

        return sample_points

    def get_sdf(self, noise_scale=0.0025):
        noise = np.random.normal(loc=0, scale=noise_scale, size=self.sample_points.shape)
        self.noisy_points = self.sample_points + noise

        self.sdf = self.mesh.nearest.signed_distance(self.noisy_points)

