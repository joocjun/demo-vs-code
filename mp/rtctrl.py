#!/usr/bin/env python3

from dataclasses import dataclass
import xmlrpc.client
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

import xmlrpc
from xmlrpc.server import SimpleXMLRPCServer
from xmlrpc.server import SimpleXMLRPCRequestHandler
from xmlrpc.client import ServerProxy



def logging_time(original_fn):
    def wrapper_fn(*args, **kwargs):
        start_time = time.time()
        result = original_fn(*args, **kwargs)
        end_time = time.time()
        print("WorkingTime[{}]: {} sec".format(original_fn.__name__, end_time-start_time))
        return result
    return wrapper_fn


class RequestHandler(SimpleXMLRPCRequestHandler):
    rpc_paths = ('/RPC2',)


@dataclass
class StateCache:
    prev_stamp: np.ndarray = np.array([1.74434292e+12])
    ps_world_right: np.ndarray = np.zeros((21,3))
    ps_world_left: np.ndarray = np.zeros((21,3))


@dataclass
class Config:
    cam: CameraConfig = CameraConfig()
    device_id: str = '043322071286'

    fps: float = 30.0
    show: bool = False

    warmup: float = 1.0

    # host: str = '137.68.192.166'
    host: str = '137.68.191.117'
    port: int = 8001

    host_in: str = 'localhost'
    port_in: int = 8002
    cam_path: str = '/tmp/cam.json'
    img_path: str = '/tmp/docker/img.png'
    out_path: str = 'out.pkl'
    vid_mode: bool = False


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

    cam_cfg = MultiRSCamera.Config.map_devices(cfg.cam,
                                               [cfg.device_id])
    predictor = ServerProxy(F'http://{cfg.host}:{cfg.port}')

    vis = None
    if cfg.show:
        vis = o3d.visualization.Visualizer()
        win = vis.create_window()

        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(0.2)
        # axis.transform(T)
        vis.add_geometry(axis)

        kpts = o3d.geometry.PointCloud()
        zero = np.zeros((21, 3))
        kpts.points = o3d.utility.Vector3dVector(zero)
        kpts.colors = o3d.utility.Vector3dVector(zero)
        vis.add_geometry(kpts)

    with MultiRSCamera(cam_cfg).open() as cam:
        
        # warm-up
        n_warmup = int(max(1, cfg.warmup / 0.025))
        for _ in range(n_warmup):
            time.sleep(0.025)
            frame = cam()
        prev_stamp = frame['stamp']

        with SimpleXMLRPCServer((cfg.host_in, cfg.port_in),
                                requestHandler=RequestHandler) as server:
            # state = {'prev_stamp': prev_stamp,
            #          'ps_world': np.zeros((21, 3))
            #          }

            state: StateCache = StateCache()

            def on_kpt(state: StateCache):
                return state.ps_world_right.tolist(), state.ps_world_left.tolist()
                # return state['ps_world'].tolist()
            server.register_introspection_functions()
            server.register_function(partial(on_kpt, state=state),
                                     'kpt')

            @logging_time
            def step(state: StateCache):
                frame = cam()

                # Skip old (or not sufficiently new) frames.
                stamp = frame['stamp']
                dt = stamp - state.prev_stamp
                
                is_new = (np.greater(dt, 1000.0 / cfg.fps).all())
                if not is_new:
                    return
                # state['prev_stamp'] = stamp
                state.prev_stamp = stamp

                # process frames.
                count: int = len(frame['stamp'])
                for j in range(count):
                    color_rgb = frame['color'][j]

                    # <- save img ->
                    if (not cfg.vid_mode):
                        cv2.imwrite(cfg.img_path, color_rgb[..., ::-1])
                    else:
                        # {vid}
                        writer = cv2.VideoWriter(
                            cfg.vid_path,
                            fourcc=cv2.VideoWriter_fourcc(
                                *"DIVX"),
                            fps=int(cfg.fps))
                        writer.write(color_rgb[..., ::-1])
                        writer.release()

                    # <- send img to HaMeR srv ->
                    if cfg.vid_mode:
                        # {vid}
                        with open(cfg.vid_path, 'rb') as fp:
                            # send_file()
                            resp = requests.post(
                                F'http://{cfg.host}:{cfg.port}',
                                files=dict(file=fp),
                                data=dict(focal=float(fx)))
                            

                        traj = resp.json()
                    else:
                        # {img}
                        color_np = color_rgb[..., ::-1]
                        binary_data = xmlrpc.client.Binary(color_np.tobytes())
                       
                         
                        out = predictor.hand_img(binary_data, 
                                                    cfg.out_path,
                                                    float(fx))

                        
                        traj = []
                        if out == 'None':
                            pass 
                        else: 
                            data = [json.loads(out)]
                            for datum in data:
                                if datum is None:
                                    continue
                                det = dict(kpt=datum['pred_keypoints_3d'],
                                        cam=datum['pred_cam_t_full'],
                                        rgt=datum['is_rights'])
                                traj.append(det)

                    
                    if len(traj) == 0:
                        return
                    
                    # <- interpret `traj` from srv ->
                    ps = []
                    rs = []
                    for det in traj:
                        c, k, r = det['cam'], det['kpt'], det['rgt']
                        c = np.asarray(c)
                        k = np.asarray(k)
                        

                        left_kpts_idx=r.index(0.0) if 0.0 in r else -1
                        right_kpts_idx=r.index(1.0) if 1.0 in r else -1 

                        for idx in [left_kpts_idx, right_kpts_idx]:
                            if idx == -1:
                                continue
                            t = c.reshape(-1,1,3)[idx]
                            p = k.reshape(-1,21,3)[idx] + t 
                            ps.append(p) 
                            rs.append(r[idx])


                        
                    # <- update `ps_world` output ->
                    for i in range(len(ps)):
                        # state['ps_world'] = (
                        #     ps[i] @ world_from_cam[: 3, : 3].T +
                        #     world_from_cam[: 3, 3]
                        # )
                        if rs[i] == 0.0: 
                            state.ps_world_left = (
                                ps[i] @ world_from_cam[: 3, : 3].T +
                                world_from_cam[: 3, 3]
                            )
                        elif rs[i] == 1.0:
                            state.ps_world_right = (
                                ps[i] @ world_from_cam[: 3, : 3].T +
                                world_from_cam[: 3, 3]
                            )
                        else: 
                            raise ValueError(f"hand side should be identified by 0.0 , 1.0 not {rs[i]}")

                        

            def loop(state):
                while True:
                    # start = time.time()
                    step(state)
                    # print('step', time.time() - start)
            # server.service_actions = partial(step, state=state)
            thread = threading.Thread(target=partial(loop, state=state),
                                      daemon=True)
            thread.start()
            server.serve_forever()
            # loop(state)


if __name__ == '__main__':
    main()
