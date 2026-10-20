"""Detachment criterion: the ONE place the rule and its threshold live.

WHY THIS MODULE EXISTS SEPARATELY
---------------------------------
The threshold is the method's one free parameter, and it is NOT yet settled:
the divergence analysis (docs/damage/NEXT_STEPS_detachment_criterion.md
§B.6i–B.6q) has still to establish what floor the det F error imposes, and
§B.6m/B.6q show Pi_1's distribution is bimodal rather than a tunable continuum.

⚠️ So the rule WILL change. Keeping it here means a future change touches one
file, not the RK4 kernel, the CLI, the npz writer and three run scripts. The
kernel receives a resolved threshold value and a mode integer; it never
encodes the policy.

THE RULE, AS CURRENTLY DEFINED
------------------------------
Buck & Langerman (2004) give a critical radius as "the location of vanishing
material velocity", a closed form in the Bingham number -- a yield-stress
group. The local statement of the same physics:

    Pi_1 = sigma_eq / tau_yield_ref          detach when Pi_1 < Pi_1_crit

sigma_eq is the local Norton flow stress; tau_yield_ref is the Norton flow
stress at a REFERENCE strain rate with the LOCAL temperature. ⚠️ The reference
is essential: on the 'models' driver sigma_eq IS the local Norton stress, so
dividing by the local value would give Pi_1 == 1 everywhere.

KINEMATICS AFTER DETACHMENT
---------------------------
A detached particle is no longer dragged by the plasticised flow, so it stops
following the velocity field. It is then carried only by the workpiece's rigid
translation past the tool -- the advancing speed v_adv, along the travel
direction. ✅ That is a definite, parameter-free choice: no new constant, and
v_adv is already read per case from data/*.som.fix.

⚠️ What this deliberately does NOT model: interaction between detached
particles. They carry no mass (they are passive tracers in a precomputed
field), so there is no force to exchange -- see §B.6e. The consequence is that
the method predicts WHERE material fails to arrive, not the void's equilibrium
shape after surrounding material consolidates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# Kinematic modes a detached particle can follow. Integers, because the value
# crosses into a JIT-traced kernel as a static Python int.
#   0 = no detachment (the rule is off)
#   1 = v_adv only: carried by the workpiece's rigid translation
#   2 = freeze: stop dead in the lab frame
#   3 = flag only: record the detachment but keep advecting normally
#       ⚠️ mode 3 is the DIAGNOSTIC mode -- it changes no trajectory, so it
#       isolates "how many detach and where" from "what detaching does".
MODE_OFF = 0
MODE_VADV = 1
MODE_FREEZE = 2
MODE_FLAG_ONLY = 3

_MODE_NAMES = {
    MODE_OFF: "off",
    MODE_VADV: "v_adv",
    MODE_FREEZE: "freeze",
    MODE_FLAG_ONLY: "flag-only",
}

MODE_FROM_NAME = {v: k for k, v in _MODE_NAMES.items()}


@dataclass(frozen=True)
class DetachmentPolicy:
    """Resolved detachment configuration handed to the kernel.

    ⚠️ Frozen on purpose: the kernel closes over these values at trace time,
    so mutating one after the kernel is built would silently not take effect.

    Attributes
    ----------
    enabled : bool
        False means the kernel compiles the detachment branch out entirely.
    pi1_crit : float
        Detach when Pi_1 < pi1_crit. ⚠️ NOT yet physically determined; see
        ``describe()`` for what is known about choosing it.
    mode : int
        One of MODE_* above: what a detached particle does afterwards.
    v_adv : float
        Advancing speed [m/s], read per case. Required for MODE_VADV.
    travel_axis : int
        Index of the travel direction (0=x, 1=y, 2=z). FSW cases travel +x.
    latch : bool
        True (default): once detached, always detached. ⚠️ Keep this True
        unless there is a physical reason for re-entrainment -- a particle
        that re-attaches every time it crosses a stress fluctuation would
        produce a void map that is really a map of threshold crossings.
    """

    enabled: bool = False
    pi1_crit: float = 0.5
    mode: int = MODE_VADV
    v_adv: Optional[float] = None
    travel_axis: int = 0
    latch: bool = True

    def validate(self) -> None:
        """Fail early and loudly, before a mesh load is paid for."""
        if not self.enabled:
            return
        if self.mode not in _MODE_NAMES:
            raise ValueError(
                f"detachment mode {self.mode} unknown; "
                f"expected one of {sorted(_MODE_NAMES)}")
        if not (0.0 < self.pi1_crit <= 10.0):
            # ⚠️ 10.0 is the clip applied to tau_ratio in
            # jaxtrace/damage/fields.py; a threshold above it can never fire.
            raise ValueError(
                f"pi1_crit must be in (0, 10]; got {self.pi1_crit}. "
                "tau_ratio is clipped at 10, so a larger threshold would "
                "detach every particle unconditionally.")
        if self.mode == MODE_VADV:
            if self.v_adv is None:
                raise ValueError(
                    "detachment mode 'v_adv' needs v_adv [m/s]. It is read "
                    "per case from data/*.som.fix; pass it explicitly if "
                    "auto-detection failed. ⚠️ Both dt and v_adv vary between "
                    "cases (O22), so a hard-coded value is wrong.")
            if not (0.0 < float(self.v_adv) < 1.0):
                raise ValueError(
                    f"v_adv = {self.v_adv} m/s is outside the plausible FSW "
                    "range (the cohort spans 0.005..0.010 m/s). A value in "
                    "mm/min was probably passed instead of m/s.")
        if self.travel_axis not in (0, 1, 2):
            raise ValueError(f"travel_axis must be 0, 1 or 2; got {self.travel_axis}")

    @property
    def mode_name(self) -> str:
        return _MODE_NAMES.get(self.mode, f"<{self.mode}>")

    def describe(self) -> str:
        """One block for the run log, so a result's rule is self-documenting."""
        if not self.enabled:
            return "  [detach] disabled"
        lines = [
            f"  [detach] rule   : Pi_1 = sigma_eq/tau_yield_ref < "
            f"{self.pi1_crit:g}  (Buck & Langerman 2004 yield-stress group)",
            f"  [detach] after  : {self.mode_name}",
        ]
        if self.mode == MODE_VADV:
            lines.append(
                f"  [detach]          carried at v_adv = {self.v_adv:g} m/s "
                f"along axis {self.travel_axis} "
                f"({'xyz'[self.travel_axis]})")
        lines.append(
            f"  [detach] latched: {self.latch}"
            + ("" if self.latch else "   ⚠️ re-entrainment allowed"))
        lines.append(
            "  [detach] ⚠️ THRESHOLD IS PROVISIONAL. tau_min is also recorded "
            "per particle, so every other threshold is a post-process on the "
            "npz -- re-running is never needed to change it.")
        return "\n".join(lines)


def policy_from_args(args, v_adv: Optional[float]) -> DetachmentPolicy:
    """Build the policy from parsed CLI args. The ONLY place args are read."""
    enabled = bool(getattr(args, "detach", False))
    mode_name = str(getattr(args, "detach_mode", "v_adv"))
    if mode_name not in MODE_FROM_NAME:
        raise ValueError(
            f"--detach-mode {mode_name!r} unknown; "
            f"expected one of {sorted(MODE_FROM_NAME)}")
    pol = DetachmentPolicy(
        enabled=enabled,
        pi1_crit=float(getattr(args, "detach_pi1_crit", 0.5)),
        mode=MODE_FROM_NAME[mode_name],
        v_adv=(float(v_adv) if v_adv is not None else None),
        travel_axis=int(getattr(args, "detach_travel_axis", 0)),
        latch=not bool(getattr(args, "detach_allow_reattach", False)),
    )
    pol.validate()
    return pol
