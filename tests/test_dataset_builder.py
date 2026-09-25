import json

import numpy as np

from generate_data import build_dataset_from_reference


def test_build_dataset_from_reference_writes_npz_and_metadata(tmp_path):
    reference_path = tmp_path / "hollywood-action.npy"
    output_path = tmp_path / "hollywood-action.npz"
    np.save(reference_path, np.ones((45, 33, 3), dtype=np.float32))

    metadata_path = build_dataset_from_reference(
        reference_path, output_path, variants_per_window=1
    )

    with np.load(output_path) as dataset:
        assert dataset["idol_seqs"].shape == (2, 30, 33, 3)
        assert dataset["user_seqs"].shape == (2, 30, 33, 3)
    assert json.loads(metadata_path.read_text(encoding="utf-8"))[0]["window_index"] == 0
