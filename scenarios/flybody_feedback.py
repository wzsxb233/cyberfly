"""Non-contact 3D markers for already delivered feedback; never inject a stimulus."""
import time
import numpy as np


class FlyBodyFeedback:
    def __init__(self):
        self.event = None

    def clear(self):
        self.event = None

    def show(self, kind='aversive', intensity=1., duration_ms=800):
        if kind not in {'aversive', 'reward', 'food'}:
            raise ValueError('Feedback marker must be aversive, reward or food.')
        if not np.isfinite(intensity) or not 0 <= intensity <= 1 or not np.isfinite(duration_ms) or not 0 < duration_ms <= 30000:
            raise ValueError('Use finite intensity [0,1] and duration (0,30000] ms.')
        self.event = {'kind': kind, 'intensity': float(intensity), 'duration_ms': float(duration_ms),
                      'expires_at_monotonic_s': time.monotonic() + duration_ms / 1000,
                      'visual_only': True, 'geometry': 'MuJoCo rendered 3D non-contact electrodes/arcs or sugar grains',
                      'neural_delivery': 'Not performed by this renderer; session invokes marker only after actual delivery'}
        return self.state()

    def state(self):
        if not self.event or self.event['expires_at_monotonic_s'] <= time.monotonic():
            return None
        return {**self.event, 'remaining_ms': max(0., (self.event['expires_at_monotonic_s'] - time.monotonic()) * 1000)}

    def draw(self, scene, body):
        import mujoco
        event = self.state()
        if event is None:
            return
        position = body.mj_data.xpos[body.thorax_body_id].copy()  # original CGS centimetres
        alpha = event['intensity']
        def geom(kind, size, pos, color, end=None):
            if scene.ngeom >= scene.maxgeom:
                raise RuntimeError('MuJoCo scene has no room for the delivered-feedback marker.')
            value = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(value, kind, np.asarray(size, float), np.asarray(pos, float),
                               np.eye(3).reshape(-1), np.asarray([*color, alpha], np.float32))
            value.category = mujoco.mjtCatBit.mjCAT_DECOR
            if end is not None:
                mujoco.mjv_connector(value, kind, float(size[0]), np.asarray(pos, float), np.asarray(end, float))
            value.emission = .7
            scene.ngeom += 1
        if event['kind'] == 'aversive':
            for side in (-1, 1):
                start = position + [side * .10, 0, .035]
                end = position + [side * .10, 0, .12]
                geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [.009, .02, .009], start, [.95, .27, .20], end)
            points = position + np.array([[-.10, 0, .12],[-.06,-.01,.145],[-.02,0,.095],
                                           [.02,-.01,.145],[.06,0,.095],[.10,0,.12]])
            for start, end in zip(points[:-1], points[1:]):
                geom(mujoco.mjtGeom.mjGEOM_CAPSULE, [.004, .01, .004], start, [1., .83, .35], end)
        else:
            for index in range(3):
                pos = position + [.09 + .025 * index, -.025 + .025 * index, .06 + .015 * (index % 2)]
                geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.018] * 3, pos,
                     [.45, 1., .68] if event['kind'] == 'reward' else [1., .83, .48])
