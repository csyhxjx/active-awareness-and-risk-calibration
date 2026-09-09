"""Read-only LIBERO constraint monitoring with onset event recording."""

import json
import os

import numpy as np


DEFAULTS = dict(
    xy_margin=0.03,
    z_min_below_table=0.02,
    z_max_above_table=0.60,
    obj_drop_below=0.03,
    gripper_penetration=0.002,
    self_col_hops=2,
)


def _get_sim(env):
    current = env
    for _ in range(5):
        if hasattr(current, "sim"):
            return current.sim
        current = current.unwrapped if hasattr(current, "unwrapped") else getattr(current, "env")
    raise AttributeError("cannot locate sim")


class LiberoConstraintMonitor:
    """Inspect simulation state without writing to the environment or observation."""

    def __init__(self, env, cfg=None, out_path=None):
        self.env = env
        self.sim = None
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self._scene_ready = False
        self._prev = {}
        self._onsets = []
        self._steps = []
        self.out_path = out_path

    def _scene(self):
        model, data = self.sim.model, self.sim.data
        bodies = [model.body_id2name(i) for i in range(model.nbody)]

        hand = next(name for name in bodies if name and name.endswith("_right_hand"))
        self.prefix = hand.rsplit("_right_hand", 1)[0]

        def body_id(name):
            return model.body_name2id(name)

        def body_geoms(body):
            return [geom for geom in range(model.ngeom) if model.geom_bodyid[geom] == body]

        table_body = next(body_id(name) for name in bodies if name and "table" in name.lower())
        table_geoms = body_geoms(table_body)
        tops = [data.geom_xpos[g][2] + model.geom_size[g][2] for g in table_geoms]
        x_bounds = [
            data.geom_xpos[g][0] - model.geom_size[g][0] for g in table_geoms
        ] + [data.geom_xpos[g][0] + model.geom_size[g][0] for g in table_geoms]
        y_bounds = [
            data.geom_xpos[g][1] - model.geom_size[g][1] for g in table_geoms
        ] + [data.geom_xpos[g][1] + model.geom_size[g][1] for g in table_geoms]
        self.table_z = max(tops)
        self.table_xy = (min(x_bounds), max(x_bounds), min(y_bounds), max(y_bounds))

        gripper_body = next(name for name in bodies if name and name.startswith("gripper0_"))
        self.gripper_prefix = gripper_body.split("_", 1)[0]
        robot_body_names = {
            name for name in bodies
            if name and (name.startswith(self.prefix) or name.startswith(self.gripper_prefix))
        }
        self.robot_geoms = {
            geom
            for name in robot_body_names
            for geom in body_geoms(body_id(name))
        }
        self.gripper_geoms = {
            geom
            for name in bodies
            if name and name.startswith(self.gripper_prefix)
            for geom in body_geoms(body_id(name))
        }

        self.object_geoms = set()
        self.object_bodies = []
        for joint in range(model.njnt):
            if model.jnt_type[joint] == 0:
                body = model.jnt_bodyid[joint]
                self.object_bodies.append(body)
                self.object_geoms.update(body_geoms(body))
        self.static_geoms = set(range(model.ngeom)) - self.robot_geoms - self.object_geoms

        self.eef_site = next(
            model.site_id2name(site)
            for site in range(model.nsite)
            if model.site_id2name(site) and "grip_site" in model.site_id2name(site)
        )

        def chain(body):
            result = set()
            while body > 0:
                result.add(body)
                body = model.body_parentid[body]
            return result

        robot_bodies = [body_id(name) for name in robot_body_names]
        self.allowed_pairs = set()
        for first in robot_bodies:
            first_chain = chain(first)
            for second in robot_bodies:
                if first == second:
                    continue
                if second in first_chain or len(first_chain & chain(second)) >= 1:
                    self.allowed_pairs.add((min(first, second), max(first, second)))

    def check(self, t):
        # LIBERO may replace its MjSim instance during reset; always resolve
        # the current simulator before reading state.
        self.sim = _get_sim(self.env)
        if not self._scene_ready:
            self._scene()
            self._scene_ready = True
        model, data = self.sim.model, self.sim.data
        eef = data.site_xpos[model.site_name2id(self.eef_site)].copy()
        x0, x1, y0, y1 = self.table_xy
        cfg = self.cfg
        rec = {"t": int(t), "eef": eef.tolist()}

        workspace_margin = min(
            eef[0] - (x0 - cfg["xy_margin"]),
            (x1 + cfg["xy_margin"]) - eef[0],
            eef[1] - (y0 - cfg["xy_margin"]),
            (y1 + cfg["xy_margin"]) - eef[1],
            eef[2] - (self.table_z - cfg["z_min_below_table"]),
            (self.table_z + cfg["z_max_above_table"]) - eef[2],
        )
        rec["workspace"] = {"violated": bool(workspace_margin < 0), "margin": float(workspace_margin)}

        gripper_penetration = 0.0
        self_collision = False
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            depth = -float(contact.dist)
            if (geom1 in self.gripper_geoms and geom2 in self.static_geoms) or (
                geom2 in self.gripper_geoms and geom1 in self.static_geoms
            ):
                gripper_penetration = max(gripper_penetration, depth)
            body1, body2 = int(model.geom_bodyid[geom1]), int(model.geom_bodyid[geom2])
            pair = (min(body1, body2), max(body1, body2))
            if (
                geom1 in self.robot_geoms
                and geom2 in self.robot_geoms
                and pair not in self.allowed_pairs
                and depth > 0
            ):
                self_collision = True

        rec["gripper_env"] = {
            "violated": gripper_penetration > cfg["gripper_penetration"],
            "margin": float(cfg["gripper_penetration"] - gripper_penetration),
        }
        rec["self_collision"] = {"violated": self_collision, "margin": 0.0}

        object_margin = 1e9
        for body in self.object_bodies:
            object_margin = min(object_margin, data.body_xpos[body][2] - (self.table_z - cfg["obj_drop_below"]))
        rec["object_drop"] = {"violated": bool(object_margin < 0), "margin": float(object_margin)}

        finite = all(
            np.isfinite(array).all()
            for array in (data.qpos, data.qvel, getattr(data, "ctrl", np.zeros(1)))
        )
        rec["non_finite"] = {"violated": not finite, "margin": 0.0 if finite else -1.0}

        for name in ("workspace", "gripper_env", "self_collision", "object_drop", "non_finite"):
            violated = rec[name]["violated"]
            if violated and not self._prev.get(name, False):
                self._onsets.append({"type": name, "t": int(t)})
            self._prev[name] = violated

        self._steps.append(rec)
        return rec

    def episode_reset(self):
        self._prev = {}
        self._onsets = []
        self._steps = []

    def finish_episode(self, task_id, trial, success):
        summary = {
            "task_id": task_id,
            "trial": trial,
            "success": bool(success),
            "num_onsets": len(self._onsets),
            "onsets": self._onsets,
            "steps": self._steps,
        }
        if self.out_path:
            os.makedirs(os.path.dirname(os.path.abspath(self.out_path)), exist_ok=True)
            with open(self.out_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary) + "\n")
        return summary
