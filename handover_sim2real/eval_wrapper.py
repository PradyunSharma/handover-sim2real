"""
Benchmark wrapper that exposes an *active-grasp* check.

`HandoverBenchmarkWrapper._check_status` computes `contact_panda_fingers` — both
Panda finger links in force-contact with the YCB object — but keeps it private.
`ycb.released` is a looser signal: it also fires on the **passive** release path
(an open gripper merely bumping the object), so it over-counts grasps in rollout
eval.

`GraspBenchmarkWrapper.grasped_active()` replicates the finger-contact test so a
rollout can report a *real* grasp (both fingers gripping the object) separately
from `released`. It reads the current contact state, so call it during/after the
grasp phase and OR it across steps.
"""

from __future__ import annotations

import numpy as np

from handover.benchmark_wrapper import HandoverBenchmarkWrapper


class GraspBenchmarkWrapper(HandoverBenchmarkWrapper):
    def grasped_active(self) -> bool:
        """True iff BOTH Panda finger links are in force-contact with the YCB
        object right now — the 'active grasp' the success check requires,
        excluding the passive collision-release path.

        Mirrors the `contact_panda_fingers` computation in the base wrapper's
        `_check_status` (object↔panda contacts in both a/b orderings, force
        thresholded, finger links collected).
        """
        contact = self.contact[0]
        if len(contact) == 0:
            return False

        ycb_id   = self.ycb.bodies[self.ycb.ids[0]].contact_id[0]
        panda_id = self.panda.body.contact_id[0]
        thresh   = self.cfg.BENCHMARK.CONTACT_FORCE_THRESH

        # Object↔panda contacts in both id orderings; the panda link is the
        # *other* body's link in each case.
        c1 = contact[(contact["body_id_a"] == ycb_id) & (contact["body_id_b"] == panda_id)]
        c2 = contact[(contact["body_id_a"] == panda_id) & (contact["body_id_b"] == ycb_id)]
        links = np.concatenate([
            c1["link_id_b"][c1["force"] > thresh],
            c2["link_id_a"][c2["force"] > thresh],
        ])
        return set(self.panda.LINK_IND_FINGERS).issubset(int(x) for x in links)
