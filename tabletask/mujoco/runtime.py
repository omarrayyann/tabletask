"""Live MuJoCo state for a task: pose queries, contacts, stepping, snapshots and rendering."""

from __future__ import annotations

import copy
import warnings
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

from ..language.rotation import mat
from ..language.schema import Task
from . import geometry
from .geometry import DIRECTIONS, collides, geoms_under, height, hull, static_contacts, under
from .model import save
from .render import DEFAULT_CAMERA, frame, make_renderer, scene_options
from .rtx import attach

SETTLE_RATE = 4  # damping per unit mass (or inertia) applied while settling, in 1/s
CHUNK = 0.05  # simulated time between stillness checks, in seconds
SETTLE_SECONDS = 8.0  # longest an object may take to come to rest after placement
SETTLED_METRES = 0.002  # max displacement over one chunk for an object to count as still
SETTLED_TURN = 0.02  # max quaternion-component change over one chunk to count as still
PAD = 0.01  # tolerance around footprints and region bounds, in metres
DROPPED = 0.1  # depth below the table top at which an object counts as fallen, in metres
UPRIGHT = 0.5  # min upward component of a contact normal for the contact to support
SEAM = 0.004  # max distance between opposing contacts that count as a pinch, in metres
MAX_REACH = 1.0  # distance cap for region overlap queries, in metres
ARTICULATIONS = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


@lru_cache(maxsize=2)  # a compiled model is ~150 MB; starts rarely repeat a scene exactly
def _compiled(xml: str) -> mujoco.MjModel:
    """Compile scene XML, cached; callers deep-copy before mutating."""
    model = mujoco.MjModel.from_xml_string(xml)

    # Attached assets do not inherit scene defaults; cap contact softness globally.
    model.geom_solref[:, 0] = np.minimum(model.geom_solref[:, 0], 0.005)
    return model


@dataclass(frozen=True)
class Snapshot:
    """Simulation state plus the model fields that change during an episode."""

    state: np.ndarray
    body_positions: np.ndarray
    body_quaternions: np.ndarray


class Runtime:
    """MuJoCo state queries, start sampling support, and physics stepping."""

    def __init__(self, task: Task, xml: str):
        self.task = task
        self.robot = None
        self.xml = xml

        self.model = copy.deepcopy(_compiled(xml))
        self.data = mujoco.MjData(self.model)
        self._cache: dict = {}
        self.started: dict = {}
        self.start_state = None
        self._renderer = None
        self._image_size = None
        self._box_renderer = None
        self._box_image_size = None
        self._rtx = None  # set by use_rtx(); None means rasterise
        # How the start came out: set by sampling (rested, spawn_penetration) and by
        # inspection.try_out (sampling_error).
        self.rested: bool | None = None
        self.spawn_penetration: float | None = None
        self.sampling_error: str | None = None

    def pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World position and wxyz quaternion of a body or region."""
        geom = self.region(name)
        if geom is not None:
            orientation = np.zeros(4)
            mujoco.mju_mat2Quat(orientation, self.data.geom_xmat[geom])
            lo, _ = self.aabb(name)
            position = self.data.geom_xpos[geom].copy()
            position[2] = lo[2]  # a region's pose sits at its floor
            return position, orientation
        body = self.body(name)
        return self.data.xpos[body].copy(), self.data.xquat[body].copy()

    def aabb(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """World bounds from visible geometry, falling back to collision geometry."""
        points = self.shape(name)
        return points.min(0), points.max(0)

    def shape(self, name: str) -> np.ndarray:
        """World points whose box is `aabb`: a region's, or an object's visible shape."""
        geom = self.region(name)
        if geom is not None:
            return self._points(geom)
        body = self.body(name)
        groups = [
            shell @ self.data.xmat[b].reshape(3, 3).T + self.data.xpos[b]
            for b, shell in self._shells(body)
        ]
        if not groups:
            bounds = self.region(f"{name}/bounds")
            groups = [
                self._points(g)
                for g in ([bounds] if bounds is not None else self._geoms_under(body))
            ]
        return np.vstack(groups)

    def extent(self, name: str) -> np.ndarray:
        """Size of an object or region along each world axis."""
        lo, hi = self.aabb(name)
        return hi - lo

    def area(self, name: str) -> float:
        """Horizontal area of an object's or region's bounding box."""
        span = self.extent(name)
        return float(span[0] * span[1])

    def within(self, name: str, point, slack: float = 0.0) -> bool:
        """Whether a world point lies in a region, or an object's bounds, grown by `slack`."""
        target = np.asarray(point, dtype=float)
        geom = self.region(name)
        mesh = geom is not None and self.model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        faces = hull(self.model, self.model.geom_dataid[geom])[0] if mesh else None
        if faces is None:
            lo, hi = self.aabb(name)
            return bool(np.all(target >= lo - slack) and np.all(target <= hi + slack))
        local = self.data.geom_xmat[geom].reshape(3, 3).T @ (target - self.data.geom_xpos[geom])
        return bool(np.all(faces[:, :3] @ local + faces[:, 3] <= slack))

    def overlap(self, a: str, b: str) -> float:
        """Signed distance between two regions, in metres; negative when they overlap.

        Uses the region geoms themselves, so the result does not depend on where
        each region's origin lies (e.g. a peg passing through a handle).

        Raises:
            KeyError: If either name is not a region.
        """
        ga, gb = self.region(a), self.region(b)
        for name, geom in ((a, ga), (b, gb)):
            if geom is None:
                raise KeyError(f"{name!r} is not a region")
        return float(mujoco.mj_geomDistance(self.model, self.data, ga, gb, MAX_REACH, None))

    def fits(self, name: str, centre, half, slack: float = 0.0) -> bool:
        """Whether a level rectangle with half-size `half` about `centre` lies in a region."""
        cx, cy, z = centre
        hx, hy = half[0], half[1]
        return all(
            self.within(name, (cx + sx * hx, cy + sy * hy, z), slack)
            for sx in (-1, 1)
            for sy in (-1, 1)
        )

    def contacts(self, a: str, b: str) -> bool:
        """Whether any geom of `a` touches any geom of `b`, inside every region named."""
        ga, gb = set(self._geoms_of(a)), set(self._geoms_of(b))
        boxes = [self.aabb(name) for name in (a, b) if self.region(name) is not None]

        def inside(where) -> bool:
            return all(
                np.all(where[:2] >= lo[:2] - PAD) and np.all(where[:2] <= hi[:2] + PAD)
                for lo, hi in boxes
            )

        contact = self.data.contact
        for c in range(self.data.ncon):
            g1, g2 = contact.geom1[c], contact.geom2[c]
            if ((g1 in ga and g2 in gb) or (g1 in gb and g2 in ga)) and inside(contact.pos[c]):
                return True
        return any(inside(where) for where in static_contacts(self.model, self.data, ga, gb))

    def fallen(self, margin: float = DROPPED) -> list[str]:
        """Non-fixed task objects more than `margin` below the table top."""
        if self.region("table/top") is None:
            return []
        surface = float(self.aabb("table/top")[0][2]) - margin
        return [
            name
            for name, obj in self.task.scene.objects.items()
            if not obj.fixed and float(self.pose(name)[0][2]) < surface
        ]

    def handled(self) -> set[str]:
        """Task objects currently in contact with the robot."""
        if self.robot is None:
            return set()
        prefix, contact, touched = self.robot.name + "/", self.data.contact, set()
        for c in range(self.data.ncon):
            names = [
                self.model.body(int(self.model.geom_bodyid[int(g)])).name
                for g in (contact.geom1[c], contact.geom2[c])
            ]
            if sum(body.startswith(prefix) for body in names) != 1:
                continue
            other = names[0] if names[1].startswith(prefix) else names[1]
            touched.add(other.split("/")[0])
        return touched & set(self.task.scene.objects)

    def resting_on(self, name: str) -> list[str]:
        """Task objects supporting `name` from below, sorted.

        Uses the normals of MuJoCo's current contacts, so no extra geometry query is needed.
        """
        mine = set(self._geoms_of(name))
        contact, holding = self.data.contact, set()
        for c in range(self.data.ncon):
            g1, g2 = int(contact.geom1[c]), int(contact.geom2[c])
            if (g1 in mine) == (g2 in mine):
                continue
            up = float(contact.frame[c][2])  # normal z, pointing from geom1 to geom2
            if (up if g2 in mine else -up) < UPRIGHT:
                continue  # sideways or downward push: not support
            body = self.model.geom_bodyid[g2 if g1 in mine else g1]
            holding.add(self.model.body(int(body)).name.split("/")[0])
        return sorted((holding & set(self.task.scene.objects)) - {name})

    def surface(self, name: str, at) -> float:
        """Height of the highest collision surface of `name` under an (x, y) point.

        For a region, the owner's geoms and static world geoms below the region's top count.
        """
        at = np.asarray(at, dtype=float)
        tops, near = [], []  # ray hits, and footprint matches the ray missed
        region = self.region(name)
        if region is None:
            candidates, ceiling = self._geoms_under(self.body(name)), None
        else:
            owner = self.model.geom_bodyid[region]
            candidates = self._geoms_under(owner) if owner else []
            candidates += [
                g
                for g in range(self.model.ngeom)
                if self.model.geom_bodyid[g] == 0 and collides(self.model, g)
            ]
            ceiling = float(self.aabb(name)[1][2])
        for g in candidates:
            if not collides(self.model, g):
                continue
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                top, hit = float(self.data.geom_xpos[g][2]), True
            else:
                points = self._points(g)
                if not under(points, at[:2], PAD):
                    continue
                top, hit = height(self.model, self.data, g, at[:2], float(points[:, 2].max()))
            if ceiling is None or top <= ceiling + PAD:
                (tops if hit else near).append(top)

        if tops or near:
            return max(tops or near)
        return float(self.pose(name)[0][2] if region is not None else self.aabb(name)[1][2])

    def velocity(self, name: str) -> np.ndarray:
        """World-frame linear then angular velocity of a body's origin, (6,)."""
        rotational_linear = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.body(name), rotational_linear, 0
        )
        return np.r_[rotational_linear[3:], rotational_linear[:3]]

    def articulation(self, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Positions, lower and upper limits of an object's hinge and slide joints, model order.

        All three are empty for a rigid object. An unlimited hinge reads wrapped to [-π, π)
        with those limits; an unlimited slide gets limits of ±1 m.
        """
        key = ("articulation", name)
        if key not in self._cache:
            body = self.body(name)
            self._cache[key] = np.array(
                [
                    j
                    for j in range(self.model.njnt)
                    if self.model.jnt_type[j] in ARTICULATIONS
                    and self.root_body(self.model.jnt_bodyid[j]) == body
                ],
                dtype=int,
            )
        joints = self._cache[key]
        positions = self.data.qpos[self.model.jnt_qposadr[joints]].copy()
        limits = self.model.jnt_range[joints].copy()
        for k, j in enumerate(joints):
            if self.model.jnt_limited[j]:
                continue
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                positions[k] = (positions[k] + np.pi) % (2 * np.pi) - np.pi
                limits[k] = (-np.pi, np.pi)
            else:
                limits[k] = (-1.0, 1.0)
        return positions, limits[:, 0], limits[:, 1]

    def grasped(self, name: str) -> bool:
        """Whether every finger of the robot touches an object; False for a fingerless robot."""
        fingers = self.robot.fingers if self.robot is not None else ()
        mine, touching = set(self._geoms_of(name)), set()
        contact = self.data.contact
        for c in range(self.data.ncon):
            g1, g2 = int(contact.geom1[c]), int(contact.geom2[c])
            other = g2 if g1 in mine else g1 if g2 in mine else None
            if other is None:
                continue
            body = int(self.model.geom_bodyid[other])
            touching.update(i for i, finger in enumerate(fingers) if body in finger)
        return bool(fingers) and len(touching) == len(fingers)

    def joint(self, name: str) -> float:
        """Current value of a named joint."""
        return float(self.data.qpos[self.model.jnt_qposadr[self._joint(name)]])

    def joint_range(self, name: str) -> tuple[float, float]:
        """The (lo, hi) limits of a named joint."""
        lo, hi = self.model.jnt_range[self._joint(name)]
        return float(lo), float(hi)

    def set_joint(self, name: str, value: float) -> None:
        """Put a joint at a value, at rest."""
        jid = self._joint(name)
        self.data.qpos[self.model.jnt_qposadr[jid]] = value
        self.data.qvel[self.model.jnt_dofadr[jid]] = 0
        self.refresh()

    @property
    def path_traced(self) -> bool:
        """Whether pictures are path-traced by RTX rather than rasterised."""
        return self._rtx is not None

    def use_rtx(self, width: int = 960, height: int = 540, *, session=None) -> bool:
        """Switch to RTX path tracing; warn and keep rasterising if unavailable.

        Returns:
            Whether path tracing is now active.
        """
        self.close_renderer()
        self._rtx = attach(self, width, height, session)
        return self._rtx is not None

    def picture(
        self, camera=DEFAULT_CAMERA, width: int = 960, height: int = 540, **kwargs
    ) -> np.ndarray:
        """Render the current state; extra keyword arguments go to the renderer in use."""
        camera = self.camera(camera) if isinstance(camera, str) else camera
        if self._rtx is not None:
            try:
                return self._rtx.frame(camera, width, height, **kwargs)
            except (RuntimeError, OSError) as error:
                self._raster_fallback(error)
                kwargs.pop("samples", None)
                kwargs.pop("skybox", None)

        return frame(self, width, height, camera=camera, **kwargs)

    def named_joints(self) -> list[str]:
        """Names of all non-free joints, which a task can drive."""
        named = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
        )
        return [
            joint
            for joint, kind in zip(named, self.model.jnt_type)
            if joint and kind != mujoco.mjtJoint.mjJNT_FREE
        ]

    def camera(self, name: str) -> str:
        """Model name of a camera: the robot's qualified name if it owns one, else `name`."""
        robot = self.robot
        return robot.qualified(name) if robot is not None and name in robot.cameras else name

    def images(self, cameras, width: int, height: int, regions=False) -> dict:
        """Render the named cameras into a dict of RGB arrays keyed by camera name.

        The renderer is reused across calls of the same size; `regions` is as in `picture`.
        """
        if self._rtx is not None:
            try:
                return self._rtx.images(cameras, width, height, regions)
            except (RuntimeError, OSError) as error:
                self._raster_fallback(error)
        if self._renderer is None or self._image_size != (width, height):
            self.close_renderer()
            self._renderer = make_renderer(self.model, width, height)
            self._image_size = (width, height)
        out = {}
        with scene_options(self, regions) as options:
            for name in cameras:
                self._renderer.update_scene(
                    self.data, camera=self.camera(name), scene_option=options
                )
                out[name] = self._renderer.render().copy()
        return out

    def bounding_boxes(self, cameras, width: int, height: int) -> dict:
        """Visible object boxes in the corresponding rasterised RGB views.

        Args:
            cameras: Camera names mapped to sequences of task object names.
            width: Image width in pixels, matching the RGB observation.
            height: Image height in pixels, matching the RGB observation.

        Returns:
            Camera-to-object mappings with int32 yxyx arrays of shape (4,) and boolean
            visible flags. Maximum coordinates are exclusive. Invisible boxes are zero.
        """
        if self._rtx is not None:
            raise ValueError("bounding boxes require rasterised RGB images")
        if self._box_renderer is None or self._box_image_size != (width, height):
            if self._box_renderer is not None:
                self._box_renderer.close()
                self._box_renderer = None
            # Disable antialiasing so object IDs cannot mix across geometry edges.
            samples = self.model.vis.quality.offsamples
            self.model.vis.quality.offsamples = 0
            try:
                self._box_renderer = make_renderer(self.model, width, height)
            finally:
                self.model.vis.quality.offsamples = samples
            self._box_renderer.enable_segmentation_rendering()
            self._box_image_size = (width, height)

        out = {}
        with scene_options(self) as options:
            for camera, names in cameras.items():
                self._box_renderer.update_scene(
                    self.data, camera=self.camera(camera), scene_option=options
                )
                segmentation = self._box_renderer.render()
                is_geom = segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
                ids = np.where(is_geom, segmentation[..., 0] + 1, 0)
                objects = {}
                for name in names:
                    key = ("box_membership", name)
                    if key not in self._cache:
                        lookup = np.zeros(self.model.ngeom + 1, dtype=bool)
                        lookup[np.asarray(self._geoms_under(self.body(name)), dtype=int) + 1] = True
                        self._cache[key] = lookup
                    mask = self._cache[key][ids]
                    rows = np.flatnonzero(mask.any(axis=1))
                    columns = np.flatnonzero(mask.any(axis=0))
                    visible = len(rows) > 0
                    box = np.zeros(4, dtype=np.int32)
                    if visible:
                        box[:] = rows[0], columns[0], rows[-1] + 1, columns[-1] + 1
                    objects[name] = {"yxyx": box, "visible": visible}
                out[camera] = objects
        return out

    def _raster_fallback(self, error):
        """Warn about an RTX failure and switch to rasterising."""
        warnings.warn(
            f"rendering with the rasteriser: RTX failed ({error})", RuntimeWarning, stacklevel=3
        )
        self.close_renderer()

    def close_renderer(self) -> None:
        """Release the RTX session and raster renderers, if open."""
        rtx, raster = self._rtx, self._renderer
        boxes = self._box_renderer
        self._rtx = None
        self._renderer = None
        self._image_size = None
        self._box_renderer = None
        self._box_image_size = None
        try:
            if rtx is not None:
                rtx.close()
        finally:
            try:
                if raster is not None:
                    raster.close()
            finally:
                if boxes is not None:
                    boxes.close()

    def reset(self) -> None:
        """Return to the scene's keyframe, if it has one."""
        mujoco.mj_resetData(self.model, self.data)
        if self.model.nkey:
            mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        if self.robot is not None:
            self.robot.reset()
        mujoco.mj_forward(self.model, self.data)

    def settle(self, max_seconds: float = SETTLE_SECONDS) -> bool:
        """Step with temporary damping until nothing moves (`run_until_still` has no damping).

        Returns:
            Whether everything came to rest within `max_seconds`.
        """
        damping = self.model.dof_damping.copy()
        free = self._free_joints()
        for j in free:
            adr = self.model.jnt_dofadr[j]
            body = int(self.model.jnt_bodyid[j])
            mass = float(self.model.body_subtreemass[body])

            # The free body is often a light wrapper; scale by the whole subtree's inertia.
            inertia = self._subtree_inertia(body)

            self.model.dof_damping[adr : adr + 3] = SETTLE_RATE * mass
            self.model.dof_damping[adr + 3 : adr + 6] = SETTLE_RATE * inertia
        try:
            return self._until_still(max_seconds, free)
        finally:
            self.model.dof_damping[:] = damping

    def advance(self, seconds: float) -> None:
        """Step physics for `seconds` of simulated time."""
        for _ in range(max(1, round(seconds / self.model.opt.timestep))):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def run_until_still(self, max_seconds: float) -> bool:
        """Run ordinary physics until nothing is moving, or the time is up."""
        return self._until_still(max_seconds, self._free_joints())

    def _free_joints(self) -> list[int]:
        """Ids of all free joints."""
        return [
            j
            for j in range(self.model.njnt)
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE
        ]

    def _until_still(self, max_seconds: float, free: list[int]) -> bool:
        """Step until no free body moves over one chunk, or the time is up.

        Thin objects on a hard table can vibrate indefinitely in place, so stillness is
        measured as displacement per chunk rather than velocity.
        """
        steps = int(CHUNK / self.model.opt.timestep)
        adrs = [self.model.jnt_qposadr[j] for j in free]
        for _ in range(max(1, int(max_seconds / CHUNK))):
            before = [self.data.qpos[a : a + 7].copy() for a in adrs]
            for _ in range(steps):
                mujoco.mj_step(self.model, self.data)
            if all(
                np.abs(self.data.qpos[a : a + 3] - was[:3]).max() <= SETTLED_METRES
                and np.abs(self.data.qpos[a + 3 : a + 7] - was[3:]).max() <= SETTLED_TURN
                for a, was in zip(adrs, before)
            ):
                return True
        return False

    def _subtree_inertia(self, body: int) -> float:
        """Approximate rotational inertia of a body's subtree about the body's origin."""
        inertia = 1e-9
        for b in range(body, self.model.nbody):
            ancestor = b
            while ancestor > body:
                ancestor = int(self.model.body_parentid[ancestor])
            if ancestor != body:
                continue
            arm = self.data.xipos[b] - self.data.xpos[body]
            inertia += float(np.mean(self.model.body_inertia[b]))
            inertia += float(self.model.body_mass[b]) * float(arm @ arm)
        return inertia

    def refresh(self, full: bool = True) -> None:
        """Refresh kinematics, or all derived quantities when full is true."""
        if full:
            mujoco.mj_forward(self.model, self.data)
        else:
            mujoco.mj_kinematics(self.model, self.data)

    def penetration(self) -> float:
        """Deepest current interpenetration between any two bodies, in metres."""
        if self.data.ncon == 0:
            return 0.0
        return float(max(0.0, -self.data.contact.dist[: self.data.ncon].min()))

    def stayed(self, name: str, region: str, slack: float = 0.03) -> bool:
        """Whether an object is still over a region, within `slack`; True if not a region."""
        if self.region(region) is None:
            return True
        lo, hi = self.aabb(region)
        over = [*self.pose(name)[0][:2], (lo[2] + hi[2]) / 2]
        return self.within(region, over, slack)

    def pinched(self, name: str) -> bool:
        """Whether an object is wedged in a narrow gap of another object.

        True when two contacts with the same other object lie within `SEAM` of each other
        and push in nearly opposite directions.
        """
        body, model, data = self.body(name), self.model, self.data
        mine = {b for b in range(model.nbody) if self.root_body(b) == body}
        held: dict[int, list] = {}  # other root body -> [(contact position, normal)]
        for i in range(data.ncon):
            contact = data.contact[i]
            body1, body2 = model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]
            other = body2 if body1 in mine else body1 if body2 in mine else None
            if other is None or other in mine or other == 0:
                continue
            normal = contact.frame[:3] if body1 in mine else -contact.frame[:3]
            for position, seen in held.setdefault(self.root_body(other), []):
                if np.linalg.norm(position - contact.pos) < SEAM and float(normal @ seen) < -0.8:
                    return True
            held[self.root_body(other)].append((contact.pos.copy(), normal.copy()))
        return False

    def root_body(self, body: int) -> int:
        """Top-level ancestor of a body (a direct child of the world body)."""
        while self.model.body_parentid[body] != 0:
            body = self.model.body_parentid[body]
        return body

    def mark_start(self) -> None:
        """Record object poses and the full state as the episode start."""
        self.started = {name: self.pose(name) for name in self.task.scene.objects}
        self.start_state = self.snapshot()

    @contextmanager
    def rewind(self):
        """Restore the start state for the duration of the block, then restore the present.

        Raises:
            RuntimeError: If `mark_start` has not been called.
        """
        if self.start_state is None:
            raise RuntimeError(
                "no start to rewind to: call mark_start() after the start has settled"
            )
        here = self.snapshot()
        try:
            self.restore(self.start_state)
            yield self
        finally:
            self.restore(here)

    def start_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Pose of an object at the episode start."""
        if name not in self.started:
            raise KeyError(f"no recorded start for {name!r}")
        return self.started[name]

    def snapshot(self) -> Snapshot:
        """Copy integration state and mutable body poses for exact restoration."""
        specification = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(self.model, specification))
        mujoco.mj_getState(self.model, self.data, state, specification)
        return Snapshot(
            state,
            self.model.body_pos.copy(),
            self.model.body_quat.copy(),
        )

    def restore(self, snapshot: Snapshot) -> None:
        """Restore simulation state and refresh derived positions and contacts."""
        self.model.body_pos[:] = snapshot.body_positions
        self.model.body_quat[:] = snapshot.body_quaternions
        mujoco.mj_setState(
            self.model, self.data, snapshot.state, mujoco.mjtState.mjSTATE_INTEGRATION
        )
        self.refresh()

    def set_pose(self, name: str, position, orientation, full: bool = True) -> None:
        """Set a body's world pose: via its free joint, or its model pose if fixed."""
        body = self.body(name)
        if name in self.task.scene.objects and self.task.scene.objects[name].fixed:
            self.model.body_pos[body] = position
            self.model.body_quat[body] = orientation
        else:
            adr = self.model.jnt_qposadr[self.model.body_jntadr[body]]
            self.data.qpos[adr : adr + 3] = position
            self.data.qpos[adr + 3 : adr + 7] = orientation
            dof = self.model.body_dofadr[body]
            self.data.qvel[dof : dof + 6] = 0
        self.refresh(full)

    def save(self, path: Path, portable: bool = False, view: str | None = None) -> None:
        """Write the current scene to `path`; see `model.save`."""
        save(self, path, portable=portable, view=view)

    def _joint(self, name: str) -> int:
        """Joint id for a name; raises KeyError if missing."""
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"no joint {name!r}")
        return jid

    def body(self, name: str) -> int:
        """Body id for an object name; raises KeyError if missing."""
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(f"no body {name!r}")
        return bid

    def region(self, name: str) -> int | None:
        """Geom id for `object/region`, or None if the name is not a region."""
        if "/" not in name:
            return None
        head, tail = name.split("/", 1)
        gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{head}/place:{tail}")
        return gid if gid >= 0 else None

    def places(self) -> list[str]:
        """Names of every region in the scene, excluding `bounds` regions."""
        named = (
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in range(self.model.ngeom)
        )
        return [
            geom.replace("/place:", "/")
            for geom in named
            if geom and "/place:" in geom and not geom.endswith("/place:bounds")
        ]

    def resolves(self, name: str) -> bool:
        """Whether a name is a body, a region or a joint."""
        if self.region(name) is not None:
            return True
        if mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0:
            return True
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) >= 0

    def _geoms_of(self, name: str) -> list[int]:
        """Geoms of an object, or the owner's and world collision geoms near a region's box."""
        if self.region(name) is None:
            key = ("geoms", name)  # cached: a body's geoms do not change during an episode
            if key not in self._cache:
                self._cache[key] = self._geoms_under(self.body(name))
            return self._cache[key]
        lo, hi = self.aabb(name)
        owner = self.model.geom_bodyid[self.region(name)]
        own = set(self._geoms_under(owner)) if owner else set()

        base = self.surface(name, (lo[:2] + hi[:2]) / 2) - PAD
        solid = []
        for g in range(self.model.ngeom):
            if not collides(self.model, g):
                continue
            if self.model.geom_bodyid[g] != 0 and g not in own:
                continue
            if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE:
                if base <= self.data.geom_xpos[g][2] <= hi[2] + PAD:
                    solid.append(g)
                continue
            points = self._points(g)
            low, high = points.min(0), points.max(0)
            if (
                np.all(low[:2] <= hi[:2] + PAD)
                and np.all(high[:2] >= lo[:2] - PAD)
                and low[2] <= hi[2] + PAD
                and high[2] >= base
            ):
                solid.append(g)
        return solid

    def _geoms_under(self, root: int) -> list[int]:
        """Geoms of a body and its descendants; see `geometry.geoms_under`."""
        return geoms_under(self.model, root)

    def _shells(self, root: int) -> list[tuple[int, np.ndarray]]:
        """Per-body hull points of a subtree's visual meshes, in body frames; cached."""
        key = ("shells", root)
        if key not in self._cache:
            local: dict[int, list[np.ndarray]] = {}
            for g in self._geoms_under(root):
                if self.model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                if collides(self.model, g):
                    continue
                corners = hull(self.model, self.model.geom_dataid[g])[1]
                local.setdefault(int(self.model.geom_bodyid[g]), []).append(
                    corners @ mat(self.model.geom_quat[g]).T + self.model.geom_pos[g]
                )
            shells = []
            for b, pieces in local.items():
                points = np.vstack(pieces)
                with suppress(Exception):
                    points = points[ConvexHull(points).vertices]

                # Keep only points extreme along some sampled direction.
                far = np.unique(np.argmax(points @ DIRECTIONS.T, axis=0))
                shells.append((b, points[far]))
            self._cache[key] = shells
        return self._cache[key]

    def _points(self, g: int) -> np.ndarray:
        """World-frame points of a geom; see `geometry.points`."""
        return geometry.points(self.model, self.data, g)
