import os
from os.path import join
import numpy as np
import json


def point_to_ray_distance(O, D, P=np.array([0., 0., 0.])):
    """
    Computes the shortest distance from point P to a ray starting at O with direction D.
    Parameters:
    P (numpy.array): 3D coordinates of the point.
    O (numpy.array): 3D coordinates of the ray's origin.
    D (numpy.array): 3D direction vector of the ray (not necessarily unit length).

    Returns:
    float: The shortest distance from P to the ray.
    """
    D = D / np.linalg.norm(D)  # Normalize direction vector
    v = P - O
    proj_length = np.dot(v, D)  # Projection scalar
    proj_point = O + proj_length * D  # Closest point on the ray
    return np.linalg.norm(P - proj_point)


class MPIIData(object):
    def __init__(self, root_path, split='test'):
        self.data = {}
        subject_idx = [str(i).zfill(2) for i in range(15)]

        for idx in subject_idx:
            label_path = join(root_path, f'p{idx}', f'p{idx}.txt')
            with open(label_path, "r") as file:
                lines = file.readlines()

            dir_path = join(root_path, f'p{idx}')

            for line in lines:
                path = line.split(' ')[0]
                full_path = join(dir_path, path)

                fc = np.array([float(line.split(' ')[-7]), float(line.split(' ')[-6]), float(line.split(' ')[-5])])
                gt = np.array([float(line.split(' ')[-4]), float(line.split(' ')[-3]), float(line.split(' ')[-2])])

                gaze_vec = gt - fc

                dist = point_to_ray_distance(O=fc, D=gaze_vec)

                # determine EC
                if dist < 35. and gaze_vec[2] < 0:

                    #print(path, dist)
                    self.data[full_path] = 1
                else:
                    self.data[full_path] = 0

    def audits(self):
        val_lst = list(self.data.values())
        total_n_ec = sum(np.array(val_lst) == 0)
        total_ec = sum(np.array(val_lst) == 1)
        total = total_ec + total_n_ec
        perc = total_ec / total
        print(f"EC samples = {total_ec} acount for {perc:.4f} total = {total}")


if __name__ == "__main__":
    root_dir = "MPIIFaceGaze"

    Dataset = MPIIData(root_dir)

    Dataset.audits()

    with open(join(root_dir, "EC_labels.json"), "w") as file:
        json.dump(Dataset.data, file, indent=4)  # Save dictionary as JSON with indentation
