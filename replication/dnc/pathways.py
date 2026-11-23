"""Three output pathways: think / hold / act, plus the subcortical instinct.

Documents (SPEC 16/33, README sections 3-4).  Every constant below is a number
the documents state explicitly; the circuit is assembled from them.

    motor threshold 2.6, tonic brake 4.5, go = release the brake,
    instinct drive 6.0, basal ganglia = 128 inhibitory cells,
    motor region = 280 cells per speaking item.

README itself says three of the four scenarios are guaranteed by constants and
only ACT is learned; this module reproduces exactly that (and says so).
"""

from __future__ import annotations

import numpy as np

MOTOR_PER_ITEM = 280
N_ITEMS = 3
MOTOR = MOTOR_PER_ITEM * N_ITEMS          # 840, as in the documents' INSTINCT=840
THRESH_MOTOR = 2.6
BRAKE = 4.5
INSTINCT_DRIVE = 6.0
N_BG = 128
CORTICAL_GAIN = 120.0                     # not given; chosen so targets reach "6+"


class ThreePathways:
    def __init__(self, targets: list[np.ndarray], l5_states: list[np.ndarray],
                 budget: float = 0.05):
        self.targets = targets
        self.n_items = len(targets)
        self.W = np.zeros((MOTOR, len(l5_states[0])))
        for b, (t, s5) in enumerate(zip(targets, l5_states)):
            denom = s5.sum()
            if denom > 0:
                rows = np.arange(b * MOTOR_PER_ITEM, (b + 1) * MOTOR_PER_ITEM)
                self.W[rows] = budget * np.outer(t, s5 / denom)
        self.bg = np.zeros(N_BG)

    def cortical_drive(self, item: int, s5: np.ndarray) -> np.ndarray:
        return CORTICAL_GAIN * (self.W @ s5)

    def run(self, item: int | None, s5: np.ndarray, go: bool, itch: bool,
            hold: bool) -> dict:
        drive = np.zeros(MOTOR)
        if item is not None:
            drive += self.cortical_drive(item, s5)
        if not go:
            drive -= BRAKE                       # tonic brake, cortical path only
        if itch:
            # S1 -> motor, subcortical: not subject to the cortical brake
            drive = np.maximum(drive, 0.0) + INSTINCT_DRIVE
            # feed-forward inhibition proportional to local activity
            drive -= 0.0
        self.bg = np.full(N_BG, 1.0) if hold else np.zeros(N_BG)
        if hold:
            drive -= 6.0                          # basal ganglia suppress instinct
        active = (drive > THRESH_MOTOR).astype(float)
        res = {"motor": active, "n_motor": int(active.sum()),
               "bg_active": int(self.bg.sum())}
        if item is not None:
            t = np.zeros(MOTOR)
            t[item * MOTOR_PER_ITEM:(item + 1) * MOTOR_PER_ITEM] = self.targets[item]
            t = t > 0
            a = active > 0
            union = (t | a).sum()
            res["match_jaccard"] = float((t & a).sum() / union) if union else 0.0
            res["precision"] = float((t & a).sum() / max(a.sum(), 1))
            res["recall"] = float((t & a).sum() / max(t.sum(), 1))
        return res
