"""Unit tests for the detachment policy -- the threshold's single source of truth.

    python3 test_detachment_policy.py

⚠️ These test the VALIDATION, not the physics. The point is that a wrong
threshold or a missing v_adv fails in the first second rather than after a
28-minute mesh load, and that the rule is recorded with every result.
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from jaxtrace.damage.detachment import (          # noqa: E402
    DetachmentPolicy, MODE_VADV, MODE_FLAG_ONLY, policy_from_args,
)


def main() -> int:
    p = DetachmentPolicy(enabled=True, pi1_crit=0.5, mode=MODE_VADV, v_adv=0.01)
    p.validate()
    print("1. valid policy accepted")
    print(p.describe())

    # ⚠️ Each of these is a mistake that would otherwise be discovered late.
    cases = [
        (dict(enabled=True, pi1_crit=0.5, mode=MODE_VADV, v_adv=None),
         "needs v_adv", "v_adv missing and auto-detection failed"),
        (dict(enabled=True, pi1_crit=0.5, mode=MODE_VADV, v_adv=600.0),
         "plausible FSW range", "mm/min passed where m/s expected"),
        (dict(enabled=True, pi1_crit=20.0, mode=MODE_VADV, v_adv=0.01),
         "clipped at 10", "threshold above the tau_ratio clip: detaches all"),
        (dict(enabled=True, pi1_crit=0.0, mode=MODE_VADV, v_adv=0.01),
         "must be in (0, 10]", "zero threshold: detaches none"),
        (dict(enabled=True, pi1_crit=0.5, mode=99, v_adv=0.01),
         "unknown", "unknown kinematic mode"),
    ]
    for kw, want, why in cases:
        try:
            DetachmentPolicy(**kw).validate()
        except ValueError as e:
            assert want in str(e), f"wrong message for {why}: {e}"
            print(f"   rejected: {why}")
        else:
            print(f"   FAILED TO REJECT: {why}")
            return 1

    # A disabled policy must not validate anything -- it is the default and
    # must never block a run that is not using detachment.
    DetachmentPolicy(enabled=False, v_adv=None, pi1_crit=-5).validate()
    print("2. disabled policy validates trivially")

    # flag-only changes no trajectory, so it needs no v_adv.
    DetachmentPolicy(enabled=True, mode=MODE_FLAG_ONLY, v_adv=None).validate()
    print("3. flag-only needs no v_adv")

    a = types.SimpleNamespace(
        detach=True, detach_pi1_crit=0.8, detach_mode="freeze",
        detach_travel_axis=0, detach_allow_reattach=False)
    q = policy_from_args(a, 0.01)
    assert q.mode_name == "freeze" and q.pi1_crit == 0.8 and q.latch
    print(f"4. from CLI args: mode={q.mode_name} crit={q.pi1_crit} "
          f"latch={q.latch}")

    # ⚠️ --detach-allow-reattach must actually clear the latch, since the
    # default is the safe direction and a silent no-op would be invisible.
    a.detach_allow_reattach = True
    assert not policy_from_args(a, 0.01).latch
    print("5. --detach-allow-reattach clears the latch")

    print("\nALL POLICY TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
