# base_controllers/chomp_slip_workspace/chomp_cost_module/__init__.py

from base_controllers.chomp_slip_workspace.chomp_cost_module.slip_energy_cost_module import (
    SlipEnergyCostModule,
)
from base_controllers.chomp_slip_workspace.chomp_cost_module.total_energy_cost_module import (
    TotalEnergyCostModule,
)
from base_controllers.chomp_slip_workspace.chomp_cost_module.terrain_geometry_cost_module import (
    TerrainGeometryCostModule,
)


def make_cost_module(cost_name: str):
    if cost_name == "slip_energy":
        return SlipEnergyCostModule()

    if cost_name == "total_energy":
        return TotalEnergyCostModule()

    if cost_name == "terrain_geometry":
        return TerrainGeometryCostModule()

    raise ValueError(f"Unknown cost module: {cost_name}")