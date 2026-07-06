"""
Orbax checkpointing for the nnx training state.

WHAT WE CHECKPOINT
    The whole `nnx.Optimizer` — which contains the model params (incl. the MoE
    router_bias nnx.Variable), the Optax optimizer state, and the step counter — as
    a single pytree. `nnx.split(optimizer)` separates the static graph definition
    from the array state; only the array state is written to disk. On restore we
    rebuild an identical optimizer skeleton, hand its state as the abstract target,
    and `nnx.update` the restored arrays back in place.

WHY A MANAGER
    Orbax's CheckpointManager gives us step-numbered checkpoints, automatic pruning
    to the last `keep` (train.keep_checkpoints), and latest-step discovery for
    resuming — all we add on top is the nnx split/merge glue.
"""

from __future__ import annotations

import os

import flax.nnx as nnx
import orbax.checkpoint as ocp


class CheckpointManager:
    def __init__(self, ckpt_dir: str, keep: int = 3):
        # Orbax requires an absolute path for its atomic-rename commits.
        self.directory = os.path.abspath(ckpt_dir)
        options = ocp.CheckpointManagerOptions(max_to_keep=keep, create=True)
        self.mgr = ocp.CheckpointManager(self.directory, options=options)

    def save(self, step: int, optimizer: nnx.Optimizer) -> None:
        """Persist the optimizer's array state at `step` (async; commits in the bg)."""
        _, state = nnx.split(optimizer)
        self.mgr.save(step, args=ocp.args.StandardSave(state))

    def restore(self, optimizer: nnx.Optimizer, step: int | None = None) -> int:
        """Restore into `optimizer` in place. Defaults to the latest checkpoint.
        Returns the step that was restored."""
        step = self.mgr.latest_step() if step is None else step
        if step is None:
            raise FileNotFoundError(f"No checkpoints found in {self.directory}")
        graphdef, abstract_state = nnx.split(optimizer)
        restored = self.mgr.restore(step, args=ocp.args.StandardRestore(abstract_state))
        nnx.update(optimizer, restored)
        return step

    def latest_step(self) -> int | None:
        return self.mgr.latest_step()

    def wait_until_finished(self) -> None:
        """Block until all pending async saves have committed (call before exit)."""
        self.mgr.wait_until_finished()

    def close(self) -> None:
        self.mgr.close()
