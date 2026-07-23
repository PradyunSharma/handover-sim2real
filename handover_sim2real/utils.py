# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA License [see LICENSE for details].

import os
import sys


def add_sys_path_from_env(name):
    assert name in os.environ, "Environment variable '{}' is not set".format(name)
    if os.environ[name] not in sys.path:
        sys.path.append(os.environ[name])


def resolve_valid_grasp_dict_path(rl_cfg, setup):
    """Resolve the CVPR2023 paper's per-scene hand-collision-filtered grasp dict.

    The paper pre-filters grasps that collide with the human hand OFFLINE (parse
    ACRONYM -> collision-check -> keep collision-free), stored as per-scene kept-
    grasp indices in `valid_grasp_dict_005.pkl` and applied by OMG at plan time
    (`OMG-Planner/omg/planner.py` `load_grasp_set`: `pose_grasp[valid_grasp_dict[
    scene_idx]]`). It is the paper's alternative to our aggressive runtime 0.08 m
    filter and retains ~716/720 s0 scenes (vs our ~351). The original `train.py`
    wires it in the same way (`cfg.omg_config["valid_grasp_dict_path"] = ...`).

    Returns an ABSOLUTE path when `RL.valid_grasp_dict_path` is set AND
    `BENCHMARK.SETUP == "s0"` (the dict is s0-specific: keys 0..719 index the s0
    scene list). Returns None otherwise. Raises if a configured file is missing so
    a typo fails fast instead of silently training without the filter.
    """
    path = rl_cfg.get("valid_grasp_dict_path")
    if not path:
        return None
    if setup != "s0":
        print(
            "[warning] RL.valid_grasp_dict_path is set but BENCHMARK.SETUP={} != s0; "
            "the dict is s0-specific (keys 0..719) -- ignoring it.".format(setup)
        )
        return None
    if not os.path.isabs(path):
        # utils.py -> handover_sim2real -> repo root. Anchoring to the package (not
        # os.getcwd()) keeps parallel workers correct regardless of their CWD.
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            "RL.valid_grasp_dict_path resolved to '{}', which does not exist.".format(path)
        )
    return path
