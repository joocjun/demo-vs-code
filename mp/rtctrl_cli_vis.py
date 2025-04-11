#!/usr/bin/env python3

from dataclasses import dataclass
from config import oc_cli
from cam import MultiRSCamera, CameraConfig
from functools import partial
import time
import cv2
import open3d as o3d
import numpy as np
import requests
import json
import pickle
import threading

from xmlrpc.client import ServerProxy


@dataclass
class Config:
    cam: CameraConfig = CameraConfig()
    device_id: str = '043322071286'

    fps: float = 30.0
    show: bool = True

    warmup: float = 1.0

    # host: str = '137.68.192.166'
    host: str = 'localhost'
    port: int = 8002
    cam_path: str = '/tmp/cam.json'


def generate_hand_skeleton():
   

    kpts = o3d.geometry.PointCloud()
    zero = np.zeros((21, 3))
    kpts.points = o3d.utility.Vector3dVector(zero)
    kpts.colors = o3d.utility.Vector3dVector(zero)

    MANO_EDGES = []
    for i in range(5):
        i0 = 1 + i * 4
        MANO_EDGES.append((0, i0))
        for j in range(3):
            MANO_EDGES.append((i0 + j, i0 + j + 1))
    MANO_EDGES = o3d.utility.Vector2iVector(MANO_EDGES)
    skel = o3d.geometry.LineSet(kpts.points, MANO_EDGES)
    return skel, kpts



@oc_cli
def main(cfg: Config):

    with open(cfg.cam_path, 'r') as fp:
        data = json.load(fp)
        data = {k: np.asarray(v, dtype=np.float32)
                for (k, v) in data.items()}
        K = data['K']
        T = data['T']  # T = cam_from_tag
        fx = K[0, 0]

        world_from_tag = np.eye(4)
        world_from_tag[:3, :3] = np.asarray([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        tag_from_cam = np.linalg.inv(T)
        world_from_cam = world_from_tag @ tag_from_cam
    proxy = ServerProxy(F'http://{cfg.host}:{cfg.port}/RPC2')


    vis = o3d.visualization.Visualizer()
    win = vis.create_window()

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
    # axis.transform(T)
    vis.add_geometry(axis)


    right_skel, right_kpts = generate_hand_skeleton()
    vis.add_geometry(right_kpts)
    vis.add_geometry(right_skel)


    left_skel, left_kpts = generate_hand_skeleton()
    vis.add_geometry(left_kpts)
    vis.add_geometry(left_skel)


    while True:
        left_out, right_out = proxy.kpt()

        if cfg.show:
            left_kpts.points = o3d.utility.Vector3dVector(
                left_out
            )
            right_kpts.points = o3d.utility.Vector3dVector(
                right_out
            )

            # app = o3d.visualization.gui.Application.instance
            # app.post_to_main_thread(win, lambda: vis.update_geometry(kpts))

            right_skel.points = right_kpts.points
            left_skel.points = left_kpts.points 
            vis.update_geometry(left_kpts)
            vis.update_geometry(left_skel)
            vis.update_geometry(right_kpts)
            vis.update_geometry(right_skel)

            for _ in range(4):
                vis.poll_events()
                vis.update_renderer()



if __name__ == '__main__':
    main()
