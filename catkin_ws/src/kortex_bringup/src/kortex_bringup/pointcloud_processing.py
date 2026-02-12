import numpy as np
import ros_numpy
import itertools
import time

import torch
import pytorch3d.ops as torch3d_ops

pc_points = 2048

def farthest_point_sampling(points, num_points=1024, use_cuda=True):
    # Expect points as numpy array [N, 4], xyz+rgb

    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
    points_tensor = torch.from_numpy(points[:, :3]).to(device)  # only xyz needed for FPS

    sampled_points, indices = torch3d_ops.sample_farthest_points(points=points_tensor.unsqueeze(0), K=[num_points])
    sampled_points = sampled_points.squeeze(0)
    indices = indices.squeeze(0)

    # Transfer only indices back to CPU (for indexing rgb)
    indices_cpu = indices.cpu().numpy()

    # Return numpy arrays: sampled xyz + indices for rgb
    return sampled_points.cpu().numpy(), indices_cpu

def vfps(points, num_points=1024, voxel_size=0.005, use_cuda=False, color=True):
    device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    # ---- 1. Voxel subsampling (Open3D) ----
    voxelized = False
    points_tensor = None
    if points.shape[0] > 20000:
        start = time.time()
        voxelized = True

        if color:
            voxel_down_pcd, voxel_point_indices, inverse_map = pcd.voxel_down_sample_and_trace(
                voxel_size,
                min_bound=points[:, :3].min(axis=0),
                max_bound=points[:, :3].max(axis=0)
            )

            voxel_xyz = np.asarray(voxel_down_pcd.points)

            while voxel_xyz.shape[0] < 15000:
                print(voxel_xyz.shape[0])
                voxel_size *= 0.9
                voxel_down_pcd, voxel_point_indices, inverse_map = pcd.voxel_down_sample_and_trace(
                    voxel_size,
                    min_bound=points[:, :3].min(axis=0),
                    max_bound=points[:, :3].max(axis=0)
                )

                voxel_xyz = np.asarray(voxel_down_pcd.points)

            voxel_indices = np.array(list(itertools.chain.from_iterable(voxel_point_indices)))  # indices into original points
        
        else:
            points_tensor = torch.from_numpy(points).float().to(device)
            coords = torch.floor(points_tensor[:, :3] / voxel_size).int()
            voxel_coords = torch.unique(coords, dim=0)
            voxel_xyz = voxel_coords.float() * voxel_size + voxel_size / 2

            while voxel_xyz.shape[0] < 15000:
                voxel_size *= 0.9
                coords = torch.floor(points_tensor[:, :3] / voxel_size).int()
                voxel_coords = torch.unique(coords, dim=0)
                voxel_xyz = voxel_coords.float() * voxel_size + voxel_size / 2

        assert voxel_xyz.shape[0] >= 1024
        points_tensor = voxel_xyz

    # ---- 2. FPS on voxelized points ----
    if points_tensor is None:
        points_tensor = torch.from_numpy(points).float().to(device)

    sampled_points, fps_idx = torch3d_ops.sample_farthest_points(points_tensor.unsqueeze(0), K=num_points)
    sampled_points = sampled_points.squeeze(0).cpu().numpy()
    if color:
        fps_idx = fps_idx.squeeze(0).cpu().numpy()

    # ---- 3. Map back to original indices for RGB/etc ----
    if color:
        if voxelized:
            sampled_indices = voxel_indices[fps_idx]
        else:
            sampled_indices = fps_idx

        return sampled_points, sampled_indices
    return [sampled_points]

def preprocess_point_cloud(points, use_cuda=True, color=True, segment=True):
    num_points = pc_points
    orientation = False

    # Convert ROS pointcloud to numpy structured array once
    pc_np = ros_numpy.numpify(points)

    xyz = np.stack([pc_np['x'], pc_np['y'], pc_np['z']], axis=-1)

    if color:
        # get the rgb part of the pointcloud
        rgb_packed = pc_np['rgb'].view(np.uint32)
        r = (rgb_packed >> 16) & 255
        g = (rgb_packed >> 8) & 255
        b = rgb_packed & 255

        rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

        points = np.concatenate([xyz, rgb], axis=-1)

        valid_mask = np.isfinite(points[:, :3]).all(axis=1)
        points = points[valid_mask]
        xyz = points[:, :3]
        rgb = points[:, 3:]
    else:
        valid_mask = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid_mask]


    if orientation:
        extrinsics_matrix = get_homogenous_matrix()
        ones = np.ones((xyz.shape[0], 1))
        xyz_h = np.hstack([xyz, ones])
        xyz_rot = xyz_h @ extrinsics_matrix.T
        xyz = xyz_rot[:, :3]

    # Crop points by workspace mask in a vectorized manner
    # Dont need to crop if segmentation
    if not segment:
        if orientation:
            WORK_SPACE = [
                [-0.4, 0.4],
                [-1.1, 1],
                [-0.4, 1]
            ]

        else:
            WORK_SPACE = [
                [-0.6, 0.3],
                [-0.4, 0.5],
                [0.2, 1]
            ]

        mask = (
            (xyz[:, 0] > WORK_SPACE[0][0]) & (xyz[:, 0] < WORK_SPACE[0][1]) &
            (xyz[:, 1] > WORK_SPACE[1][0]) & (xyz[:, 1] < WORK_SPACE[1][1]) &
            (xyz[:, 2] > WORK_SPACE[2][0]) & (xyz[:, 2] < WORK_SPACE[2][1])
        )
        xyz = xyz[mask]
    if xyz.shape[0] == 0:
        raise ValueError("No points after cropping.")

    if color:
        rgb = rgb[mask]

        # Normalize rgb to [0, 1]
        rgb = rgb.astype(np.float32) / 255.0

        # Stack xyz and rgb into one array [N, 6]
        points_combined = np.hstack([xyz, rgb])

    # Run farthest point sampling (using xyz only)
    sampled = vfps(xyz, num_points, use_cuda=use_cuda, color=color) # return: (points, indices) if color else (points)
    points_xyz = sampled[0]

    # Adjust offsets if orientation enabled
    if orientation:
        points_xyz[..., :3] -= [-0.04489961, -0.6327338, -0.34466678]
    
    if color:
        sampled_rgb = rgb[sampled[1]]

    # Combine sampled XYZ and RGB for final result
    if color:
        return np.concatenate([points_xyz, sampled_rgb], axis=-1)
    return points_xyz


def get_homogenous_matrix():
	# Tall Cam
	rx_deg = 37  # Rotation around X
	ry_deg = 180  # Rotation around Y
	rz_deg = 0  # Rotation around Z

	# Convert to radians
	rx = np.radians(rx_deg)
	ry = np.radians(ry_deg)
	rz = np.radians(rz_deg)

	# Rotation matrix around X-axis
	Rx = np.array([
		[1, 0,          0,           0],
		[0, np.cos(rx), -np.sin(rx), 0],
		[0, np.sin(rx), np.cos(rx),  0],
		[0, 0,          0,           1]
	])

	# Rotation matrix around Y-axis
	Ry = np.array([
		[np.cos(ry),  0, np.sin(ry), 0],
		[0,           1, 0,          0],
		[-np.sin(ry), 0, np.cos(ry), 0],
		[0,           0, 0,          1]
	])

	# Rotation matrix around Z-axis
	Rz = np.array([
		[np.cos(rz), -np.sin(rz), 0, 0],
		[np.sin(rz),  np.cos(rz), 0, 0],
		[0,           0,          1, 0],
		[0,           0,          0, 1]
	])

	# Original extrinsics matrix (identity in this case)
	extrinsics_matrix = np.eye(4)

	# Combine rotations (Z * Y * X) — typical convention (can change based on your coordinate system)
	rotation_combined = Rz @ Ry @ Rx

	# Apply rotation to extrinsics
	rotated_extrinsics = rotation_combined @ extrinsics_matrix

	return rotated_extrinsics
