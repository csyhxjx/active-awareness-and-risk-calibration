"""Side-effect-free JSONL recorder for policy and environment actions."""

import os

import numpy as np

from guard.json_io import append_jsonl


class ParityRecorder:
    def __init__(self, out_path):
        self.out_path = out_path
        self.enabled = out_path is not None
        if self.enabled:
            os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def _write(self, record):
        append_jsonl(self.out_path, record)

    def log_chunk(self, task_id, trial, inference_idx, action_chunk):
        if self.enabled:
            self._write(
                {
                    "type": "chunk",
                    "task_id": task_id,
                    "trial": trial,
                    "inference_idx": inference_idx,
                    "action_chunk": np.asarray(action_chunk).tolist(),
                }
            )

    def log_step(self, task_id, trial, inference_idx, chunk_idx, env_action):
        if self.enabled:
            self._write(
                {
                    "type": "step",
                    "task_id": task_id,
                    "trial": trial,
                    "inference_idx": inference_idx,
                    "chunk_idx": chunk_idx,
                    "env_action": np.asarray(env_action).tolist(),
                }
            )

    def log_episode(self, task_id, trial, success, total_steps, num_inferences):
        if self.enabled:
            self._write(
                {
                    "type": "episode",
                    "task_id": task_id,
                    "trial": trial,
                    "success": bool(success),
                    "total_steps": total_steps,
                    "num_inferences": num_inferences,
                }
            )
