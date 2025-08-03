import os
import numpy as np
import open3d as o3d
import laspy

def read_point_cloud(file_path):

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".ply":
        pcd = o3d.io.read_point_cloud(file_path)
        return np.asarray(pcd.points).astype(np.float32)
    
    elif ext == '.xyz':
        pcd = o3d.io.read_point_cloud(file_path, format='xyz')
        return np.asarray(pcd.points).astype(np.float32)
    
    elif ext == '.las':
        las = laspy.read(file_path)
        points = np.vstack([las.x, las.y, las.z]).transpose().astype(np.float32)
        return points

    else:
        raise ValueError(f"Unsupported file format: {ext}")
    


if __name__ == "__main__":
    path = ""  
    points = read_point_cloud(path)