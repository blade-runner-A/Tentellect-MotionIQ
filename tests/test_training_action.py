"""Tests for action training command assembly."""

from __future__ import annotations

import argparse

from src.training.train_action import build_train_command


def test_build_train_command_includes_core_and_optional_flags() -> None:
    args = argparse.Namespace(
        mmaction_train_script="tools/train.py",
        config="configs/action_stgcn.py",
        work_dir="models/checkpoints/action",
        resume_from="models/checkpoints/action/last.pth",
        cfg_options=["train_cfg.max_epochs=50", "optim_wrapper.lr=0.01"],
    )

    command = build_train_command(args)

    assert command[0].endswith("python") or command[0].endswith("python.exe")
    assert command[1] == "tools/train.py"
    assert "--work-dir" in command
    assert "--resume" in command
    assert "--cfg-options" in command
