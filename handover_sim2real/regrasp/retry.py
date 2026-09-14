"""
The regrasp retry state machine: which direction to try next, and when to stop.

    loop:
        recompute the anchor frame          (the hand may have moved)
        available = ALL - attempted - neighbours(attempted, 30 deg)
        if available is empty or attempts >= cap:  SIGNAL_HUMAN, stop
        d = the available direction furthest from everything attempted
        roll the policy conditioned on d
        on failure: mark d, attempts += 1

RANKING IS ANGULAR DISTANCE ONLY. No hand-clearance term, no reachability prior,
no weights. That is a deliberate choice about what the experiment can conclude:
with a tuned ranker, a good `chained_retry_at_k` could be the ranker finding the
one feasible direction rather than the policy following instruction. With
argmax-min-angle there is nothing to tune, so any gain is attributable to the
policy. A clearance term is the obvious upgrade AFTER the conditioning is shown
to work, not before.

`attempted` IS A LIST OF WORLD-FRAME UNIT VECTORS, NEVER BIN INDICES, and this is
the subtlest requirement in the module. The anchor frame is rebuilt from the live
wrist and object centroid every step, so it rotates as the human moves. A bin
INDEX is a name in that rotating frame: bin 2 before the human turns and bin 2
after are different physical directions. Storing indices would make the exclusion
set silently drift onto directions that were never tried, and re-admit ones that
were. Storing world vectors and re-projecting them through the current anchor at
every decision is what keeps "already attempted" meaning the same thing
throughout an episode.

That failure is INVISIBLE in the current sim configuration, which is exactly why
it is written down here: under `YCB_MANO_START_FRAME: last` the hand is static,
the anchor never rotates, and indices and vectors agree perfectly. It appears on
a moving-hand config and on hardware.

THE STOP CONDITION IS A RESULT, NOT AN ERROR. `SIGNAL_HUMAN` means the robot has
exhausted its hypotheses and should ask the person to re-present the object. That
is the correct terminal behaviour for a handover system and should be reported as
its own outcome, not folded into "failure".

MEASURED FEASIBILITY BOUNDS WHAT THIS CAN DO. On s0/train, `-z` (from beneath) is
reachable by 0 of 623 scenes and `-x` (over the giver's fingers) by 12, with 11
demonstrations in the whole dataset. So the ladder has FOUR live rungs and
`chained_retry_at_k` saturates at k=4; commanding the other two is extrapolation
into directions the policy never saw. Pass `feasible=` to exclude them per scene.

Pure numpy. No simulator, no torch — the machine is unit-testable without a
rollout, which is the point of keeping it out of `chained_retry.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from handover_sim2real.regrasp import directions as D

SIGNAL_HUMAN = "SIGNAL_HUMAN"
EXHAUSTED = "EXHAUSTED"          # every direction excluded
CAP_REACHED = "CAP_REACHED"      # attempt budget spent


@dataclass
class RetryParams:
    """Knobs for the ladder. Deliberately few."""

    max_attempts: int = 4
    """Attempt cap. 4 rather than 6 because only four bins are demonstrable on
    this dataset; raising it does not create hypotheses, it only lets the machine
    command directions the policy never learned."""

    neighbour_deg: float = 30.0
    """Exclusion radius around an attempted direction. For the octahedral set
    (90 deg apart) this removes only the attempted bin itself, so it is insurance
    for a denser `k` rather than something doing work at k=6."""

    bins: object = None
    """Direction set; defaults to the octahedral `directions.BINS`."""

    prior: object = None
    """Optional bin preference order, best first — e.g. `(4, 0, 3, 2)`.

    WITHOUT IT the ladder ranks geometrically: `furthest_from` picks the
    surviving bin at the greatest angle from everything already attempted, which
    maximises the contrast between hypotheses and is deliberately blind to which
    direction is more likely to work. That is the right default for a machine
    with no history — it is the choice that makes any gain in
    `chained_retry_at_k` attributable to the policy rather than to a tuned
    ranker, which is the argument `README_REGRASP.md` makes for it.

    WITH IT the ladder tries the named bins in the named order, still subject to
    the same exclusion of `neighbour_deg` around everything attempted and to
    `feasible`. Bins absent from `prior` sort behind the ones in it, ranked
    geometrically among themselves, so a partial order never shortens the
    ladder. This is what `eval_regrasp_testset.py --adaptive` feeds in: an order
    learned online from per-bin success, which is what a deployment would have
    and the geometric rule cannot express.

    The two rules answer different questions and neither is a fix for the other.
    Geometric maximises information per attempt; a success prior maximises the
    chance the FIRST attempt works. On a robot facing a person holding an object
    the second is what matters; in an experiment measuring whether direction
    conditioning does anything, the first is."""

    def resolve_bins(self) -> np.ndarray:
        return D.BINS if self.bins is None else np.asarray(self.bins, dtype=np.float64)

    def resolve_prior(self):
        """`{bin: rank}` or None. Lower rank = tried earlier."""
        if self.prior is None:
            return None
        return {int(b): r for r, b in enumerate(self.prior)}


@dataclass
class RetryState:
    """One episode's ladder. Construct per scene, step per attempt."""

    attempted_world: list = field(default_factory=list)
    """WORLD-frame unit vectors, one per attempt made. See the module docstring
    for why this is not a list of bin indices."""

    attempts: int = 0
    outcomes: list = field(default_factory=list)     # (bin_idx, reason) per attempt
    stopped: str | None = None

    def mark(self, d_world, bin_idx: int, reason: str = "FAIL") -> None:
        """Record an attempt that did not succeed."""
        self.attempted_world.append(
            D.normalize(np.asarray(d_world, dtype=np.float64)))
        self.outcomes.append((int(bin_idx), str(reason)))
        self.attempts += 1


def next_direction(state: RetryState, anchor_R, params: RetryParams | None = None,
                   feasible=None):
    """(d_world, bin_idx, info) for the next attempt, or (None, None, info) to stop.

    `anchor_R` is the CURRENT anchor rotation — recomputed by the caller each
    time, because the frame tracks the human. `feasible` optionally restricts the
    bins to those this scene can actually reach; without it the machine will
    happily command `-z`, which no scene in this dataset affords.
    """
    params = params or RetryParams()
    bins = params.resolve_bins()
    R = np.asarray(anchor_R, dtype=np.float64)

    if state.attempts >= params.max_attempts:
        state.stopped = CAP_REACHED
        return None, None, {"stop": SIGNAL_HUMAN, "why": CAP_REACHED,
                            "attempts": state.attempts}

    # Project the attempted WORLD directions into the CURRENT anchor frame before
    # comparing. This is the line the whole module is arranged around: it is what
    # makes "already attempted" survive the frame rotating under it.
    attempted_anchor = [D.from_world(d, R) for d in state.attempted_world]

    available = D.exclude(attempted_anchor, params.neighbour_deg, bins)
    if feasible is not None:
        allowed = {int(b) for b in feasible}
        available = [b for b in available if b in allowed]

    if not available:
        state.stopped = EXHAUSTED
        return None, None, {"stop": SIGNAL_HUMAN, "why": EXHAUSTED,
                            "attempts": state.attempts}

    # A SUCCESS PRIOR OVERRIDES THE GEOMETRIC RANK, and only the rank — the
    # exclusion above still runs, so a prior cannot re-command a direction
    # already attempted or one within `neighbour_deg` of it. Bins the prior does
    # not name fall to the back and keep the geometric order among themselves,
    # so a partial prior extends the ladder rather than truncating it.
    prior = params.resolve_prior()
    if prior is None:
        pick = D.furthest_from(attempted_anchor, available, bins)
    else:
        back = len(prior)
        geo = D.furthest_from(attempted_anchor, available, bins)
        pick = min(available,
                   key=lambda b: (prior.get(int(b), back), 0 if b == geo else 1,
                                  int(b)))
    d_world = D.to_world(bins[pick], R)
    margin = (float(min(D.angle_between(bins[pick], a) for a in attempted_anchor))
              if attempted_anchor else float("nan"))
    return d_world, int(pick), {"stop": None, "attempts": state.attempts,
                                "available": available, "margin_deg": margin}


def ladder(anchor_R, params: RetryParams | None = None, feasible=None) -> list:
    """The full sequence a scene WOULD be tried in, assuming every attempt fails.

    Offline planning aid: it needs no rollouts, so the order can be inspected and
    argued about before any evaluation. A live retry must still call
    `next_direction` per attempt, because the anchor moves and the order can
    change under it.
    """
    st = RetryState()
    out = []
    while True:
        d, b, info = next_direction(st, anchor_R, params, feasible)
        if d is None:
            break
        out.append((int(b), d))
        st.mark(d, b)
    return out
