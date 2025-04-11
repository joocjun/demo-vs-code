#!/usr/bin/env python3

import time
from dataclasses import dataclass, replace
from typing import Tuple, Iterable, Optional
import numpy as np
from contextlib import contextmanager

import pyrealsense2 as rs


@dataclass
class CameraConfig:
    device_id: str = '043322071286'
    img_width: int = 640
    img_height: int = 480
    fps: int = 30
    depth_scale: float = 1000.0
    cloud: bool = False
    exposure: Optional[int] = None


class MultiRSCamera:
    @dataclass
    class Config:
        cams: Tuple[CameraConfig, ...] = ()
        cloud: bool = False
        global_time: bool = False
        filter: bool = False

        @property
        def dev_ids(self) -> Tuple[str, ...]:
            return [c.device_id for c in self.cams]

        @classmethod
        def map_devices(cls,
                        cfg: CameraConfig,
                        dev: Iterable[str],
                        *args,
                        **kwds):
            cams = tuple([replace(cfg, device_id=d) for d in dev])
            return cls(cams, *args, **kwds)

    def __init__(self, cfg: Config, start: bool = True):
        self.cfg = cfg

        self.configs = []
        self.pipelines = []
        self.queues = []
        self.aligns = []
        self.frames = []
        self.filters = []

        # Configure devices.
        if True:
            dev_ids = [cam.device_id for cam in cfg.cams]
            ctx = rs.context()

            serials = [dev.get_info(rs.camera_info.serial_number)
                    for dev in ctx.devices]
            has_all = set(dev_ids).issubset(serials)
            if not has_all:
                raise ValueError(F'{dev_ids} not all in {serials}')

            for i in range(len(ctx.devices)):
                sn = ctx.devices[i].get_info(rs.camera_info.serial_number)
                if sn not in dev_ids:
                    continue

                s = ctx.devices[i].first_depth_sensor()
                print(ctx.devices[i])
                print(s)

                # disable global time
                if not cfg.global_time:
                    s.set_option(rs.option.global_time_enabled, 0)

                # mode = 1 if is_master else 2
                # s.set_option(rs.option.inter_cam_sync_mode, mode)
                # s.set_option(rs.option.output_trigger_enabled, 1)
                # s.set_option(rs.option.frames_queue_size, 2)
                try:
                    if cfg.cams[i].exposure is None:
                        print('auto-exposure')

                        s.set_option(rs.option.enable_auto_exposure, 1)
                    else:
                        exposure = cfg.cams[i].exposure
                        print(F'manual-exposure @ {exposure}')

                        s.set_option(rs.option.enable_auto_exposure, 0)
                        s.set_option(rs.option.exposure, exposure)
                except BaseException:
                    continue

        for cam in cfg.cams:
            pipeline = rs.pipeline()

            rs_cfg = rs.config()
            rs_cfg.enable_device(cam.device_id)

            filter = None
            if cfg.filter:
                filter = rs.temporal_filter()
                rs_cfg.enable_stream(rs.stream.depth, cam.img_width,
                                     cam.img_height, rs.format.z16, cam.fps)
            else:
                rs_cfg.enable_stream(rs.stream.depth, cam.img_width,
                                     cam.img_height, rs.format.z16, cam.fps)
            rs_cfg.enable_stream(rs.stream.color, cam.img_width,
                                 cam.img_height, rs.format.rgb8, cam.fps)

            queue = rs.frame_queue(2)
            align = rs.align(rs.stream.color)

            self.configs.append(rs_cfg)
            self.pipelines.append(pipeline)
            self.queues.append(queue)
            self.aligns.append(align)
            self.frames.append(None)
            self.filters.append(filter)

        self.pipe_profs = None
        self.Ks = None

        if start:
            self.start()

    @property
    def num_cam(self):
        return len(self.cfg.cams)

    @property
    def devices(self):
        return [cam.device_id for cam in self.cfg.cams]

    def start(self):

        self.pipe_profs = [
            p.start(c, q)
            for (p, c, q) in zip(
                self.pipelines,
                self.configs,
                self.queues)]
        streams = [prof.get_stream(rs.stream.color)
                   for prof in self.pipe_profs]
        intrinsics = [stream.as_video_stream_profile().get_intrinsics()
                      for stream in streams]
        self.intrinsics = intrinsics
        self.Ks = [np.asarray([
            [i.fx, 0, i.ppx],
            [0, i.fy, i.ppy],
            [0, 0, 1]], dtype=np.float32)
            for i in intrinsics]
        self.Ks = np.asarray(self.Ks, dtype=np.float32)

    def wait(self, delay: float = 0.01):
        while True:
            outputs = self()
            if outputs is not None:
                break
            time.sleep(delay)
        return outputs

    def shutdown(self):
        [p.stop() for p in self.pipelines]

    @contextmanager
    def open(self):
        try:
            # self.reset() ??
            # self.start()
            self.wait()
            yield self
        finally:
            self.shutdown()

    def __call__(self):
        cfg = self.cfg

        # frames = [q.wait_for_frame() for q in self.queues]
        frames = [q.poll_for_frame().as_frameset() for q in self.queues]
        self.frames = [
            f0 if (
                f1.size() <= 0) else a.process(
                f1.as_frameset()) for (
                a,
                f0,
                f1) in zip(
                    self.aligns,
                    self.frames,
                frames)]

        frames = self.frames
        if any([f is None for f in frames]):
            return None

        # == processing the frames ==
        colors = [np.asanyarray(f.get_color_frame().get_data())
                  for f in frames]
        if cfg.filter:
            depths = [
                np.array(
                    rs.spatial_filter().process(
                        X.process(
                            f.get_depth_frame())).get_data()) /
                cfg.depth_scale for (
                    cfg,
                    f,
                    X) in zip(
                        self.cfg.cams,
                        frames,
                    self.filters)]
        else:
            depths = [np.array(
                f.get_depth_frame().get_data()) / cfg.depth_scale
                for (cfg, f) in zip(self.cfg.cams, frames)]
        stamps = [f.get_timestamp() for f in frames]

        out = {
            'color': colors,
            'depth': depths,
            'stamp': stamps
        }
        out = {k: np.stack(v, axis=0) for (k, v) in out.items()}

        if cfg.cloud:
            cloud = [rs.pointcloud() for _ in frames]
            cloud = [
                pcd.calculate(
                    f.get_depth_frame()) for (
                    pcd,
                    f) in zip(
                    cloud,
                    frames)]
            cloud = [np.asanyarray(pcd.get_vertices()).view(np.float32)
                     for pcd in cloud]
            cloud = np.stack(cloud, axis=0)
            out['cloud'] = cloud
        return out


if __name__ == '__main__':
    from config import oc_cli
    import cv2

    @dataclass
    class Config:
        cam: CameraConfig = CameraConfig()
        dev: Optional[Tuple[str, ...]] = None

    @oc_cli
    def main(cfg: Config):
        if cfg.dev is None:
            # list devices
            ctx = rs.context()
            print(' == list of connected devices == ')
            for dev in ctx.devices:
                print(dev.get_info(rs.camera_info.serial_number))
            return

        cam_cfg = MultiRSCamera.Config.map_devices(cfg.cam, cfg.dev)
        with MultiRSCamera(cam_cfg, start=True).open() as cam:
            frame = cam()
            print('ddd')
            cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
            while True:
                frame = cam()
                cv2.imshow('frame', frame['color'].squeeze(axis=0)[..., ::-1])
                k = cv2.waitKey(1)
                if (k & 0xFF) == ord('q'):
                    break

    main()
