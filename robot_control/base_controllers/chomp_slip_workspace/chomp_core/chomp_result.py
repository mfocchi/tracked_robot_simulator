# base_controllers/chomp_slip_workspace/chomp_core/chomp_result.py

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class ChompResult:
    xi_world: np.ndarray
    xi_meters: np.ndarray

    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray

    v: np.ndarray
    omega: np.ndarray

    dt: float

    cost_history: List[float] = field(default_factory=list)
    trajectory_history_m: Optional[List[np.ndarray]] = None

    cost_name: str = ""
    gradient_name: str = ""