"""Default population priors for synthetic hidden truth, per calculator.

TODO(physician-review): all values below are rough, plausible ED populations for each
calculator's intended use, chosen so every decision category occurs. They are not estimates
from any dataset. Parameters are sampled independently (except DBP < SBP). The PERC
population is a low-pretest-probability one, matching PERC's intended use.
"""

from calc_bounds.distributions import Bernoulli, Categorical, Distribution, TruncNormal

B = Bernoulli
C = Categorical
N = TruncNormal

DEFAULT_PRIORS: dict[str, dict[str, Distribution]] = {
    "heart": {  # ED chest pain
        "heart_history": C(probs=(0.45, 0.35, 0.20)),
        "heart_ecg": C(probs=(0.60, 0.30, 0.10)),
        "age": N(mean=58, sd=15, lo=18, hi=100),
        "hypertension": B(p=0.40),
        "hypercholesterolemia": B(p=0.30),
        "diabetes": B(p=0.20),
        "obesity": B(p=0.25),
        "smoking": B(p=0.25),
        "family_history_cad": B(p=0.20),
        "atherosclerotic_disease": B(p=0.20),
        "heart_troponin": C(probs=(0.75, 0.15, 0.10)),
    },
    "curb65": {  # community-acquired pneumonia
        "confusion": B(p=0.15),
        "urea": N(mean=7.0, sd=3.5, lo=1.5, hi=40, decimals=1),
        "resp_rate": N(mean=22, sd=6, lo=8, hi=50),
        "sbp": N(mean=125, sd=25, lo=60, hi=220),
        "dbp": N(mean=72, sd=14, lo=30, hi=130),
        "age": N(mean=65, sd=17, lo=18, hi=100),
    },
    "qsofa": {  # suspected infection
        "resp_rate": N(mean=21, sd=5, lo=8, hi=50),
        "altered_mentation": B(p=0.20),
        "sbp": N(mean=118, sd=24, lo=60, hi=220),
    },
    "perc": {  # suspected PE, low gestalt pretest probability
        "age": N(mean=38, sd=12, lo=18, hi=90),
        "heart_rate": N(mean=88, sd=15, lo=45, hi=160),
        "spo2": N(mean=97, sd=2, lo=85, hi=100),
        "unilateral_leg_swelling": B(p=0.05),
        "hemoptysis": B(p=0.03),
        "recent_surgery_trauma": B(p=0.05),
        "prior_vte": B(p=0.05),
        "hormone_use": B(p=0.12),
    },
    "wells_pe": {  # suspected PE
        "dvt_signs": B(p=0.12),
        "pe_most_likely": B(p=0.30),
        "heart_rate": N(mean=95, sd=18, lo=45, hi=170),
        "immobilization_or_surgery": B(p=0.12),
        "prior_vte": B(p=0.12),
        "hemoptysis": B(p=0.05),
        "malignancy": B(p=0.10),
    },
    "cockcroft_gault": {  # adults needing renal dosing
        "age": N(mean=60, sd=17, lo=18, hi=100),
        "weight": N(mean=75, sd=17, lo=35, hi=180, decimals=1),
        "creatinine": N(mean=1.3, sd=0.8, lo=0.4, hi=8.0, decimals=2),
        "sex": C(probs=(0.5, 0.5)),
    },
}
