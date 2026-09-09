"""Configuration for synthetic WTB 3D damage data generation."""
from dataclasses import dataclass


@dataclass
class GenConfig:
    num_scenes: int = 1000
    points: int = 20000           # points per scan
    n_ref: int = 20000            # dense CAD reference points (shared across scenes)
    # blade geometry
    n_sections: int = 40
    n_airfoil: int = 48
    span: float = 5.0             # m
    c_root: float = 1.2           # m
    c_tip: float = 0.4            # m
    twist_deg: float = 8.0
    # sensor simulation
    noise_mm: float = 0.2         # std
    dropout: float = 0.15
    occlusion: float = 0.0        # >0 enables single-side (visible) view
    max_angle_deg: float = 0.3    # scan misregistration
    max_trans_mm: float = 5.0
    # ground truth
    gt_eps_mm: float = 0.6        # |deviation| above this (mm) = damaged
    seed: int = 0
    out_dir: str = "data"

    @property
    def gt_eps(self):
        return self.gt_eps_mm * 1e-3
