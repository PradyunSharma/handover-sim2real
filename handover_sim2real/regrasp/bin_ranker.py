"""
Online per-bin success ranking — which direction to command FIRST, learned as
the sweep runs.

WHAT PROBLEM THIS SOLVES. `directions.RETRY_LADDER` is a FIXED order
(`+x, +z, +y, -y, -z, -x`) hardcoded from run 11's measured per-bin success. It
is baked into `_regrasp_metrics`, so `retry_at_k` on every run since 16 answers
"what would retrying buy, given run 11's opinion of which direction is best".
That is a defensible constant for comparing runs to each other, and the wrong
constant for asking what a DEPLOYMENT would achieve: a deployment does not know
run 11, it knows what has worked on the objects it has seen so far.

This module is the second reading. It keeps a Beta-Bernoulli posterior over each
bin's success probability, updates it after every episode, and hands back the
order the next scene should be tried in. Feed it the whole test split scene by
scene and `retry_at_k` becomes "success within k attempts, where the attempt
ORDER was itself learned online, using only scenes already seen".

CAUSALITY IS THE WHOLE POINT, AND IT IS EASY TO LOSE. The ranker must be updated
only from episodes that have already been scored, and the order used for scene i
must be the order standing BEFORE scene i is rolled out. Fit the ranking on the
whole sweep and then re-reduce, and `retry_at_k` silently becomes an oracle: it
would be ordering by a success rate computed partly from the very episodes it is
about to score. `BinRanker` cannot be used that way by accident — `order()`
reads the posterior, `observe()` writes it, and the caller interleaves them per
episode — but it is worth naming, because the offline version is one `groupby`
away and looks identical in a plot.

WHAT IT CHANGES, AND WHAT IT CANNOT
    INDEPENDENT SWEEP. Every (scene, bin) pair is rolled out regardless of order,
        so per-bin rates, `dir_err` and the outcome taxonomy are ORDER-INVARIANT
        and do not move. The ranker changes exactly one thing: the reduction
        `retry_at_k` is taken over. Do not read an improved `adaptive_retry_at_k`
        as a better policy — it is the same rollouts, ordered better.
    CHAINED SWEEP / --stop-on-success. Here the order decides which rollouts
        HAPPEN at all (attempt 2 runs only if attempt 1 failed), so a good
        ranking genuinely costs fewer attempts. `mean_attempts` is the number
        that moves, and it is a real saving rather than a re-reduction.

THE PRIOR IS THE PASSED SEQUENCE. `--bins +z,+x,-y,+y` is not a filter, it is
the ranking to start from: bin r in the order gets a Beta prior centred at a
`p0` ramped linearly from `prior_hi` down to `prior_lo`, with total pseudo-count
`prior_strength`. Before any data the posterior means are monotone in that order,
so the first scene is tried exactly as asked; after roughly `prior_strength`
episodes per bin the evidence dominates. One mechanism, no tie-break epsilon,
and the strength of the opinion is a number the user sets rather than a property
of the sort.

MODES, and why the default is not the obvious one.
    fixed      never updates. The passed sequence, held. The control condition,
               and the thing to diff `ucb` against.
    mean       rank by posterior mean. What "rank the successful bin higher"
               literally says, and the one to avoid at small n: a bin that goes
               1-for-1 early sits at the top with a posterior std of 0.2, and
               greedy ranking never pays the cost of finding that out.
    ucb        posterior mean + `ucb_c` posterior standard deviations. DEFAULT.
               A bin that has been tried twice is uncertain, so it keeps a bonus
               and keeps getting sampled; a bin tried eighty times is trusted at
               its mean. This is the version that can be defended as a
               deployment policy rather than as a plot.
    thompson   sample each bin's rate from its posterior and rank the sample.
               Randomised, so it needs `seed` to be reproducible, and it makes
               two runs of the same sweep give different ladders. Offered
               because it is the textbook answer; `ucb` is deterministic, which
               matters more here — a test-set number that moves when you re-run
               it is a test-set number nobody trusts.

NOT A BANDIT IN THE USUAL SENSE. A bandit chooses ONE arm and collects reward.
Here the independent sweep pulls every arm on every scene, so there is no
exploration cost to pay and `ucb` is doing bookkeeping rather than managing a
regret tradeoff. The cost only becomes real under `--chained` or
`--stop-on-success`, where a scene stops at its first success and the bins ranked
last genuinely go unsampled — which is also exactly when the per-bin denominators
stop being comparable across bins. That asymmetry is reported, not hidden: see
`n_pulled` in `report()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from handover_sim2real.regrasp import directions as _D

MODES = ("fixed", "mean", "ucb", "thompson")


def parse_bin_sequence(spec, *, live=None) -> list:
    """`'+z,+x,-y'` or `'4,0,3'` -> `[4, 0, 3]`. Names come from `BIN_SHORT`.

    Accepts the short names the figures and the console print (`+x`, `-z`), the
    long ones from `BIN_NAMES` (`+x_free_end`), and bare indices, because all
    three appear in this repo's own output and a user pasting from any of them
    should not have to translate. Unknown tokens raise with the legal set spelled
    out rather than being dropped — a silently ignored bin would shorten the
    ladder without saying so.

    `live` (optional) restricts the result and, when the spec is empty, SUPPLIES
    it: `parse_bin_sequence(None, live=table_bins)` is how a caller says "the
    default order over whatever this table can reach".
    """
    short = {s: i for i, s in enumerate(_D.BIN_SHORT[:len(_D.BINS)])}
    long = {n: i for i, n in enumerate(_D.BIN_NAMES)}
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        base = list(_D.RETRY_LADDER)
    elif isinstance(spec, str):
        base = []
        for tok in spec.replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok in short:
                b = short[tok]
            elif tok in long:
                b = long[tok]
            else:
                try:
                    b = int(tok)
                except ValueError:
                    raise SystemExit(
                        f"[rank] cannot read {tok!r} as a bin. Use the short "
                        f"names {list(_D.BIN_SHORT[:len(_D.BINS)])}, the long "
                        f"names {list(_D.BIN_NAMES)}, or indices "
                        f"0..{len(_D.BINS) - 1}.") from None
            if not 0 <= b < len(_D.BINS):
                raise SystemExit(f"[rank] bin {b} out of range "
                                 f"0..{len(_D.BINS) - 1}")
            if b not in base:
                base.append(b)
    else:
        base = [int(b) for b in spec]

    if live is not None:
        allowed = {int(b) for b in live}
        kept = [b for b in base if b in allowed]
        # Bins the split cannot realise are dropped; live bins the user did not
        # name are APPENDED behind the ones they did. That is what makes
        # `--bins` a PREFERENCE rather than a filter — naming three of four bins
        # ranks those three first and still evaluates the fourth. `--only-bins`
        # is the filter, and it is a separate flag for exactly this reason.
        for b in sorted(allowed):
            if b not in kept:
                kept.append(b)
        base = kept
    return base


@dataclass
class BinRanker:
    """Beta-Bernoulli posterior per bin, ranked. Construct once per sweep.

    `prior_order` is the sequence to start from; everything else is the shape of
    the opinion held about it. The posterior for bin b is
    `Beta(a0_b + successes_b, b0_b + failures_b)` with
    `a0_b = prior_strength * p0_b`, `p0_b` ramped over `prior_order`.
    """

    prior_order: tuple = ()
    mode: str = "ucb"
    prior_strength: float = 6.0
    """Total pseudo-count of the prior, in EPISODES. 6 means the passed sequence
    is worth about six observations per bin — enough that one lucky scene does
    not reorder the ladder, weak enough that 129 test scenes overrule it."""
    prior_hi: float = 0.60
    prior_lo: float = 0.40
    """The `p0` ramp across the passed order. The SPREAD is what encodes the
    opinion; the centre is irrelevant because only the ranking is read. Narrow
    on purpose: a 0.6-to-0.4 ramp at strength 6 puts the first and last bin
    1.2 pseudo-successes apart, which one real episode can close."""
    ucb_c: float = 1.0
    """Posterior standard deviations added in `ucb` mode."""
    seed: int = 0

    succ: dict = field(default_factory=dict)
    fail: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    """(episode_index, bin, success) — the full update stream, so the ladder's
    evolution can be replayed and plotted without re-running the sweep."""

    def __post_init__(self):
        if self.mode not in MODES:
            raise SystemExit(f"[rank] mode must be one of {list(MODES)}, "
                             f"got {self.mode!r}")
        if not self.prior_order:
            self.prior_order = tuple(_D.RETRY_LADDER)
        self.prior_order = tuple(int(b) for b in self.prior_order)
        self._rng = np.random.default_rng(int(self.seed))
        n = max(len(self.prior_order) - 1, 1)
        # p0 ramps hi -> lo across the passed order. A bin absent from it (an
        # impossible sequence, but cheap to survive) sits at the bottom.
        self._p0 = {b: (self.prior_hi
                        + (self.prior_lo - self.prior_hi) * r / n)
                    for r, b in enumerate(self.prior_order)}
        self._rank0 = {b: r for r, b in enumerate(self.prior_order)}

    # ---- posterior -------------------------------------------------------
    def _ab(self, b: int):
        p0 = self._p0.get(int(b), self.prior_lo)
        a = self.prior_strength * p0 + self.succ.get(int(b), 0)
        bb = self.prior_strength * (1.0 - p0) + self.fail.get(int(b), 0)
        return float(a), float(bb)

    def posterior_mean(self, b: int) -> float:
        a, bb = self._ab(b)
        return a / (a + bb)

    def posterior_sd(self, b: int) -> float:
        a, bb = self._ab(b)
        t = a + bb
        return float(np.sqrt(a * bb / (t * t * (t + 1.0))))

    def score(self, b: int) -> float:
        """Higher = try earlier. `fixed` returns the negated prior rank so the
        one code path below sorts every mode identically."""
        b = int(b)
        if self.mode == "fixed":
            return -float(self._rank0.get(b, len(self.prior_order)))
        if self.mode == "mean":
            return self.posterior_mean(b)
        if self.mode == "ucb":
            return self.posterior_mean(b) + self.ucb_c * self.posterior_sd(b)
        a, bb = self._ab(b)                       # thompson
        return float(self._rng.beta(a, bb))

    # ---- the two operations the caller interleaves -----------------------
    def order(self, feasible=None) -> list:
        """The bins to try, best first. Ties break on the PASSED order, never on
        the bin index — an index tie-break would quietly reinstate `+x, -x, +y`
        whenever two bins were level, which is the arbitrary order the fixed
        ladder exists to replace."""
        bs = ([int(b) for b in feasible] if feasible is not None
              else list(self.prior_order))
        return sorted(set(bs),
                      key=lambda b: (-self.score(b),
                                     self._rank0.get(b, len(self.prior_order))))

    def observe(self, b: int, success: bool, *, episode: int = -1) -> None:
        b = int(b)
        if success:
            self.succ[b] = self.succ.get(b, 0) + 1
        else:
            self.fail[b] = self.fail.get(b, 0) + 1
        self.history.append((int(episode), b, int(bool(success))))

    # ---- reporting -------------------------------------------------------
    def n_pulled(self, b: int) -> int:
        return int(self.succ.get(int(b), 0) + self.fail.get(int(b), 0))

    def empirical(self, b: int) -> float:
        """Raw successes/attempts, NO prior. Reported alongside the posterior
        because they answer different questions: the posterior is what the
        ranker acted on, this is what the data says. A large gap on a bin means
        the ladder was still being driven by the prior there."""
        n = self.n_pulled(b)
        return self.succ.get(int(b), 0) / n if n else float("nan")

    def report(self) -> dict:
        """Flat, CSV-ready: `rank_order`, and per bin the posterior, the raw
        rate, and the pull count."""
        order = self.order()
        out = {"rank_mode": self.mode,
               "rank_order": "|".join(_D.BIN_SHORT[b] for b in order),
               "rank_order_idx": "|".join(str(b) for b in order)}
        for r, b in enumerate(order):
            out[f"rank_pos_b{b}"] = r
        for b in range(len(_D.BINS)):
            out[f"rank_post_b{b}"] = round(self.posterior_mean(b), 4)
            out[f"rank_emp_b{b}"] = ("" if self.n_pulled(b) == 0
                                     else round(self.empirical(b), 4))
            out[f"rank_n_b{b}"] = self.n_pulled(b)
        return out

    def describe(self) -> str:
        order = self.order()
        parts = []
        for b in order:
            n = self.n_pulled(b)
            parts.append(f"{_D.BIN_SHORT[b]} {self.posterior_mean(b):.2f}"
                         + (f"({self.succ.get(b, 0)}/{n})" if n else "(prior)"))
        return f"[{self.mode}] " + "  ".join(parts)
