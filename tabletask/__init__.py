"""Tabletop manipulation tasks written as JSON, sampled and graded in MuJoCo."""

from .env import Env
from .inspection import preview as _preview
from .language.errors import SamplingError, UnsupportedError
from .language.format import dumps, json_schema, load, tasks
from .language.grammar import words
from .language.kernel import relation, term
from .language.schema import Task
from .sampling import instantiate
from .vector import Envs

__version__ = "0.2.2"

__all__ = [
    "Env",
    "Envs",
    "SamplingError",
    "Task",
    "UnsupportedError",
    "dumps",
    "instantiate",
    "json_schema",
    "load",
    "make",
    "preview",
    "relation",
    "tasks",
    "term",
    "words",
]


def make(
    source,
    *,
    num_envs=None,
    robot="droid",
    gpus=None,
    renderer=None,
    reward="success",
    bounding_boxes=None,
    scene=None,
    bindings=None,
    monitor_interval=0.01,
    backend="mujoco",
) -> Env | Envs:
    """Make an environment for a task, or several that run in parallel.

    Args:
        source: A task id from `tasks()`, a path to a task file, a JSON string, or a dict.
        num_envs: If given, make this many environments, each in its own process, and
            return them as one `Envs`.
        robot: The robot that acts in the scene: "droid", "gripper", a `Robot`, or None.
        gpus: With `num_envs`, CUDA GPU indices the environments render on, in turn.
        renderer: None to rasterise images, or "rtx" to path-trace them.
        reward: "success" for 1.0 on success, or "subtasks" for 1.0 at each new milestone.
        bounding_boxes: Optional camera-to-object-names mapping, such as
            {"external": ["item1"]} for the pick-max target. Labels use visible pixels
            in the returned RGB images. Defaults to None, disabling label rendering.
        scene: A scene XML file to use in place of the task's own.
        bindings: Assets for abstract objects, as {name: {"asset": path, "properties": {...}}}.
        monitor_interval: How often, in simulated seconds, task conditions are checked.
        backend: The simulator; only "mujoco" is provided.

    Returns:
        An `Env`, or with `num_envs` an `Envs`, ready to `reset()`.
    """
    if num_envs is not None:
        return Envs(
            source,
            num_envs,
            robot=robot,
            gpus=gpus,
            renderer=renderer,
            reward=reward,
            bounding_boxes=bounding_boxes,
            scene=scene,
            bindings=bindings,
            monitor_interval=monitor_interval,
            backend=backend,
        )
    if gpus is not None:
        raise ValueError("gpus go with num_envs")
    env = Env(load(source), scene, bindings, monitor_interval=monitor_interval, backend=backend)
    env.renderer = renderer
    env.reward = reward
    env.bounding_boxes = bounding_boxes or {}
    if isinstance(robot, str):
        robot = _robot(robot)
    return env if robot is None else env.add(robot)


def _robot(name):
    from . import robots

    names = {"droid": "Droid", "gripper": "Gripper"}
    if name not in names:
        raise ValueError(f"unknown robot {name!r}; choose from {sorted(names)}")
    return getattr(robots, names[name])()


def preview(source, samples=10, out=None, **kwargs):
    """Render a grid of sampled starts for a task.

    Args:
        source: A task id, task file, JSON string or dict.
        samples: How many starts to draw, from episodes 0, 1, ...
        out: Where to save the picture, if anywhere.
        **kwargs: `columns`, `renderer="rtx"`, or picture options such as `width`.

    Returns:
        The grid as a PIL image.
    """
    return _preview(load(source), samples, out, **kwargs)
