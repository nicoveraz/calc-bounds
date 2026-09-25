"""Question-asking policies. Each returns a full `Trace`.

S1 ask-all · S2 LLM agent · S3 bounds · S4 bounds + VOI + confidence echo · S3-bin ablation.
"""

from calc_bounds.policies.ask_all import AskAllPolicy
from calc_bounds.policies.base import Policy, PolicyId, Step, Trace
from calc_bounds.policies.bounds_policy import BoundsPolicy

POLICIES: dict[str, Policy] = {
    "s1_ask_all": AskAllPolicy(),
    "s3_bounds": BoundsPolicy(binary=False),
    "s3_bin": BoundsPolicy(binary=True),
}

__all__ = ["POLICIES", "AskAllPolicy", "BoundsPolicy", "Policy", "PolicyId", "Step", "Trace"]
