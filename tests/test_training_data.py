import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from dataset import combine_dca_npz_datasets, create_group_split_indices


def _write_dataset(path: Path, score: float):
    sample_count = 2
    np.savez_compressed(
        path,
        idol_seqs=np.zeros((sample_count, 30, 33, 3), dtype=np.float32),
        user_seqs=np.ones((sample_count, 30, 33, 3), dtype=np.float32),
        scores=np.full(sample_count, score, dtype=np.float32),
        joint_errors=np.zeros((sample_count, 33), dtype=np.float32),
        highlight_joints=np.zeros((sample_count, 3), dtype=np.int64),
    )
    path.with_name(f"{path.stem}_metadata.json").write_text(
        json.dumps([{"sample_id": 0, "window_index": 0}, {"sample_id": 1, "window_index": 1}]),
        encoding="utf-8",
    )


class TrainingDataTest(unittest.TestCase):
    def test_combined_datasets_keep_song_group_ids_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.npz"
            second = root / "second.npz"
            third = root / "third.npz"
            _write_dataset(first, 10.0)
            _write_dataset(second, 90.0)
            _write_dataset(third, 50.0)

            data, group_ids = combine_dca_npz_datasets([first, second, third])
            split = create_group_split_indices(group_ids, seed=7)

            self.assertEqual(data["scores"].tolist(), [10.0, 10.0, 90.0, 90.0, 50.0, 50.0])
            self.assertEqual(group_ids, ["first", "first", "second", "second", "third", "third"])
            for indices in split.values():
                self.assertEqual(len(indices), 2)


if __name__ == "__main__":
    unittest.main()
