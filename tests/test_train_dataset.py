from pathlib import Path
import tempfile
import unittest

from train_dataset import find_dataset_paths


class TrainDatasetTest(unittest.TestCase):
    def test_find_dataset_paths_returns_sorted_npz_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "zeta.npz").touch()
            (root / "alpha.npz").touch()
            (root / "ignored.npy").touch()

            self.assertEqual(
                find_dataset_paths(root),
                [root / "alpha.npz", root / "zeta.npz"],
            )


if __name__ == "__main__":
    unittest.main()
