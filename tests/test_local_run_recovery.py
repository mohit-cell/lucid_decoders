from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from lucid_decoders.tools import local_status
from lucid_decoders.tools.local_run import (
    acquire_lock,
    append_event,
    new_state,
    release_lock,
    save_state,
    write_heartbeat,
)


class LocalRunRecoveryTests(unittest.TestCase):
    def test_state_lock_and_heartbeat_files_are_recoverable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts_dir = root / "artifacts"
            logs_dir = artifacts_dir / "logs"
            logs_dir.mkdir(parents=True)
            state_path = artifacts_dir / "run_state.json"
            lock_path = logs_dir / "run.lock"

            args = build_args(root)
            state = new_state(args)
            append_event(state, "test", "created")
            save_state(state, state_path)
            acquire_lock(lock_path, state, logs_dir)
            write_heartbeat(logs_dir, state, "env-check", pid=123)

            self.assertTrue(state_path.exists())
            self.assertTrue(lock_path.exists())
            self.assertTrue((logs_dir / "heartbeat.jsonl").exists())

            release_lock(lock_path)
            self.assertFalse(lock_path.exists())

    def test_status_reports_resume_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = build_args(root)
            status = local_status.collect_status(
                Namespace(
                    run_id=args.run_id,
                    processed_dir=args.processed_dir,
                    artifacts_dir=args.artifacts_dir,
                    model_types=["logistic_regression"],
                    persist_head_models="best",
                )
            )

            self.assertIn("lucid_decoders.tools.local_run", status["resume_command"])
            self.assertEqual(status["run_id"], "test_run")


def build_args(root: Path) -> Namespace:
    return Namespace(
        run_id="test_run",
        processed_dir=str(root / "processed"),
        artifacts_dir=str(root / "artifacts"),
        normalized_source=str(root / "source.jsonl"),
        model_name="fake",
        source_lang="en_XX",
        target_lang="de_DE",
        device="cpu",
        chunk_size=2,
        seed=13,
        train_per_label=2,
        validation_per_label=1,
        test_per_label=1,
        model_types=["logistic_regression"],
        head_train_jobs=1,
        persist_head_models="best",
    )


if __name__ == "__main__":
    unittest.main()
