"""Offline checks against the vendored bridge's fixed PD gain contract."""

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

_BRIDGE_SOURCE = (
  Path(__file__).resolve().parents[2]
  / "third-party/openhomie/HomieDeploy/unitree_sdk2/g1_control.cpp"
)


def load_lcm_bridge_pd_gains(
  source_path: Path = _BRIDGE_SOURCE,
) -> tuple[np.ndarray, np.ndarray]:
  """Read the C++ gain literals and convert to the bridge's float32 wire type.

  This verifies the checked-out source contract; the bridge executable must
  have been rebuilt from that same source revision.
  """
  try:
    source = source_path.read_text()
    source = re.sub(r"/\*.*?\*/|//[^\n]*", "", source, flags=re.DOTALL)
    count_match = re.search(r"\bG1_NUM_MOTOR\s*=\s*(\d+)\s*;", source)
    if count_match is None:
      raise ValueError("missing G1_NUM_MOTOR declaration")
    count = int(count_match.group(1))
    gains = []
    for name in ("Kp", "Kd"):
      match = re.search(
        rf"std::array<float,\s*G1_NUM_MOTOR>\s+{name}\s*\{{([^}}]+)\}}\s*;",
        source,
      )
      if match is None:
        raise ValueError(f"missing fixed {name} array")
      values = np.array(
        [float(value) for value in match.group(1).split(",") if value.strip()],
        dtype=np.float32,
      )
      if values.shape != (count,) or not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError(f"{name} must contain {count} finite positive gains")
      gains.append(values)
    return gains[0], gains[1]
  except (OSError, ValueError) as exc:
    raise ValueError(
      f"Cannot verify fixed LCM bridge gains from {source_path}: {exc}"
    ) from exc


def validate_lcm_pd_gains(
  stiffnesses: np.ndarray,
  dampings: np.ndarray,
  joint_names: Sequence[str],
) -> None:
  """Reject gains that would differ from the fixed bridge's float32 commands."""
  bridge_gains = load_lcm_bridge_pd_gains()
  for name, configured, expected in zip(
    ("Kp", "Kd"), (stiffnesses, dampings), bridge_gains, strict=True
  ):
    values = np.asarray(configured, dtype=np.float64)
    if values.shape != expected.shape or len(joint_names) != len(expected):
      raise ValueError(f"LCM {name} requires {len(expected)} gains in hardware order")
    if not np.all(np.isfinite(values) & (values > 0)):
      raise ValueError(f"LCM {name} requires finite positive gains")
    mismatches = np.flatnonzero(values.astype(np.float32) != expected).tolist()
    if mismatches:
      details = ", ".join(
        f"{joint_names[index]}: policy={values[index]:.9g}, bridge={expected[index]:.9g}"
        for index in mismatches
      )
      raise ValueError(
        f"Policy {name} does not match the fixed LCM bridge ({details}). "
        "Gain overrides are unsupported: the bridge does not receive Kp/Kd over LCM."
      )
