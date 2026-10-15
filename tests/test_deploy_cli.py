"""Deployment argument parsing without constructing controllers or interfaces."""

import importlib
import sys
from types import ModuleType

import pytest
from deploy.base_controller import BaseController


@pytest.fixture(
  params=[
    "deploy_wtw",
    "deploy_velocity",
    "deploy_force_control",
    "deploy_jumping_track",
  ]
)
def launcher(
  request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> ModuleType:
  def forbid_controller_init(*args, **kwargs):
    raise AssertionError("Argument parsing must not construct a controller")

  monkeypatch.setattr(BaseController, "__init__", forbid_controller_init)
  return importlib.import_module(f"deploy.{request.param}")


def test_policy_path_is_required(
  launcher: ModuleType,
  monkeypatch: pytest.MonkeyPatch,
  capsys: pytest.CaptureFixture[str],
) -> None:
  monkeypatch.setattr(sys, "argv", [launcher.__name__])

  with pytest.raises(SystemExit) as error:
    launcher.parse_args()

  assert error.value.code == 2
  assert "--policy" in capsys.readouterr().err


def test_explicit_policy_path_is_accepted_without_loading_it(
  launcher: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
  policy_path = "my exported policies/policy.jit"
  monkeypatch.setattr(sys, "argv", [launcher.__name__, "--policy", policy_path])

  args = launcher.parse_args()

  assert args.policy == policy_path
  assert args.interface == "mujoco"
