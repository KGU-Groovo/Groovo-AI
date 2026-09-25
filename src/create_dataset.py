import argparse
from pathlib import Path

import numpy as np

from generate_data import build_dataset_from_reference


def main():
    parser = argparse.ArgumentParser(description="댄스 영상에서 Groovo 학습용 NPZ를 만든다.")
    parser.add_argument("--video", required=True, help="기준 댄스 영상 경로")
    parser.add_argument("--pose-model", help="MediaPipe Tasks 런타임에서만 필요한 .task 경로")
    parser.add_argument("--song-id", required=True, help="출력 파일 이름에 사용할 song ID")
    parser.add_argument("--output-dir", default="data", help="출력 디렉터리")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--variants-per-window", type=int, default=10)
    args = parser.parse_args()

    from pose_extraction import extract_pose33_from_video

    output_dir = Path(args.output_dir)
    poses, skipped_frames = extract_pose33_from_video(
        args.video, args.pose_model, args.sample_every
    )
    reference_path = output_dir / f"{args.song_id}.npy"
    output_path = output_dir / f"{args.song_id}.npz"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(reference_path, poses)
    metadata_path = build_dataset_from_reference(
        reference_path, output_path, variants_per_window=args.variants_per_window
    )
    print(f"reference: {reference_path}")
    print(f"dataset: {output_path}")
    print(f"metadata: {metadata_path}")
    print(f"skipped pose frames: {len(skipped_frames)}")


if __name__ == "__main__":
    main()
