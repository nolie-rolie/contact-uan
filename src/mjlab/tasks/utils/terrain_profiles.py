"""Terrain presets used by the release G1 locomotion tasks."""

from __future__ import annotations

import mjlab.terrains as terrain_gen
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

BUMPY_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.25),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.75,
      noise_range=(0.02, 0.05),
      noise_step=0.02,
      border_width=0.25,
    ),
  },
  add_lights=True,
)
