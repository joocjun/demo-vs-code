#!/usr/bin/env python3

from typing import Optional
from natsort import natsorted
from pathlib import Path
from dataclasses import dataclass
import pickle
import json
import cv2
from config import oc_cli
import numpy as np
import open3d as o3d


@dataclass
class Config:
    seq_path: str = '/tmp/sav002/'
    traj_path: str = '/tmp/sav002/traj.pkl'
    cam_path: str = '/tmp/sav002/cam.json'
    show_cloud: bool = False


def get_images(cfg: Config):
    colors = []
    depths = []

    for i, f in enumerate(natsorted(Path(cfg.seq_path).glob('color*.png'))):
        f_c = str(f)
        color = cv2.imread(f_c, cv2.IMREAD_COLOR)[..., ::-1]

        f_d = str(f).replace('color', 'depth')
        depth = cv2.imread(f_d, cv2.IMREAD_UNCHANGED) / 1000.0
        colors.append(color)
        depths.append(depth)
    colors = np.stack(colors, axis=0)
    depths = np.stack(depths, axis=0)
    return (colors, depths)


def np2o3d_img2pcd(
        color: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
        T_cb: Optional[np.ndarray] = None,
        normal: bool = True,
        depth_max: float = 3.0
):
    """
    Args:
        color: color image
        depth: depth image
        K: intrinsics
        T_cb: extrinsics ("camera_from_base" transform)
    """
    T_cb = o3d.core.Tensor.from_numpy(T_cb)
    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(
        o3d.t.geometry.RGBDImage(
            o3d.t.geometry.Image(color.astype(np.uint8)),
            o3d.t.geometry.Image(depth.astype(np.float32))
        ),
        K,
        depth_scale=1.0,
        with_normals=normal,
        extrinsics=T_cb,
        depth_max=depth_max
    )
    return pcd.to_legacy()


def pose_from_kpt(k: np.ndarray) -> np.ndarray:
    """
    Arg:
        k: A[..., 21, 3] MANO keypoint array.

    Return:
        A[..., 4,4] homogeneous matrix representing the tool pose
        (world_from_tool transform) formatted as [R,t; 0,1]
    """
    wrist = k[..., 0, :]
    index = k[..., 8, :]
    thumb = k[..., 4, :]
    u1 = index - wrist
    u2 = thumb - wrist

    z = np.cross(u1, u2)
    z /= np.linalg.norm(z, axis=-1, keepdims=True)

    x = 0.5 * (u1 + u2)
    x /= np.linalg.norm(x, axis=-1, keepdims=True)

    y = np.cross(z, x)
    y /= np.linalg.norm(y, axis=-1, keepdims=True)

    z = np.cross(x, y)
    z /= np.linalg.norm(y, axis=-1, keepdims=True)

    R = np.stack([x, y, z], axis=-1)
    T = [np.eye(4) for _ in range(len(k))]
    T = np.stack(T, axis=0)
    T[..., :3, :3] = R
    T[..., :3, 3] = 0.5 * (index + thumb)
    return T


@oc_cli
def main(cfg: Config):
    # Load camera intrinsics & extrinsics.
    with open(cfg.cam_path, 'r') as fp:
        data = json.load(fp)
        data = {k: np.asarray(v, dtype=np.float32)
                for (k, v) in data.items()}
        K = data['K']
        T = data['T']  # T = cam_from_tag

    # load computed trajectory (output of client.py)
    try:
        with open(cfg.traj_path, 'rb') as fp:
            traj = pickle.load(fp)
    except FileNotFoundError:
        traj=None

    # optionally show the point cloud.
    if cfg.show_cloud:
        color, depth = get_images(cfg)
        clouds = [np2o3d_img2pcd(c, d, K,
                                 np.eye(4)) for (c, d) in
                  zip(color, depth)]
        
    # apply camera transform to `kpt`
    ps = []
    if traj is not None:
        for c, k, r in zip(traj['cam'],
                        traj['kpt'],
                        traj['rgt']):
            c = np.asarray(c)
            k = np.asarray(k)
            t = c.reshape(-1, 1, 3)[-1]
            p = k.reshape(-1, 21, 3)[-1] + t
            ps.append(p)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # axis shows where the tag is (was) w.r.t. the camera
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
    axis.transform(T)
    vis.add_geometry(axis)

    if traj is not None:
        pose = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
        vis.add_geometry(pose)
        Ts = pose_from_kpt(np.stack(ps, axis=0))

        # Hand keypoints visualization
        kpts = o3d.geometry.PointCloud()
        kpts.points = o3d.utility.Vector3dVector(ps[0])
        kpts.colors = o3d.utility.Vector3dVector(np.zeros_like(ps[0]))
        vis.add_geometry(kpts)

    if cfg.show_cloud:
        cloud = o3d.geometry.PointCloud()
        vis.add_geometry(cloud)

    for j in range(100000):

        if traj is not None:
            i = j % len(ps)
            p = o3d.geometry.TriangleMesh.create_coordinate_frame(0.1)
            p.transform(Ts[i])
            pose.vertices = p.vertices
            vis.update_geometry(pose)

            # keypoints
            kpts.points = o3d.utility.Vector3dVector(ps[i])
            vis.update_geometry(kpts)
        else:
            i=0

        # point cloud (optional)
        if cfg.show_cloud:
            cloud.points = clouds[i].points
            cloud.colors = clouds[i].colors
            vis.update_geometry(cloud)
        for _ in range(4):
            vis.poll_events()
            vis.update_renderer()


if __name__ == '__main__':
    main()
