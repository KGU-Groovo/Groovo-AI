from pathlib import Path
import json
import re
import numpy as np
import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def extract_number_from_filename(path):
    path = Path(path)
    numbers = re.findall(r"\d+", path.stem)

    if numbers:
        return (0, int(numbers[-1]), path.name)

    return (1, path.name)


def list_image_files(image_dir, recursive=False):
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")

    exts = {".jpg", ".jpeg", ".png"}

    if recursive:
        files = [p for p in image_dir.rglob("*") if p.suffix.lower() in exts]
    else:
        files = [
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in exts
        ]

    files = sorted(files, key=extract_number_from_filename)

    if len(files) == 0:
        raise FileNotFoundError(f"No image files found in: {image_dir}")

    return files


def create_pose_landmarker(model_path):
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"PoseLandmarker model file not found: {model_path}")

    base_options = python.BaseOptions(
        model_asset_path=str(model_path),
        delegate=python.BaseOptions.Delegate.CPU,
    )

    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    detector = vision.PoseLandmarker.create_from_options(options)
    return detector


def extract_pose33_from_image(image_path, detector):
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"image file does not exist: {image_path}")

    image_bgr = cv2.imread(str(image_path))

    if image_bgr is None:
        raise ValueError(f"cv2 failed to read image: {image_path}")

    return extract_pose33_from_bgr_frame(image_bgr, detector)


def extract_pose33_from_bgr_frame(image_bgr, detector):
    """BGR 비디오 프레임 하나에서 MediaPipe Pose 33개 좌표를 추출한다."""

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=image_rgb,
    )

    result = detector.detect(mp_image)

    if result.pose_landmarks is None or len(result.pose_landmarks) == 0:
        return None

    landmarks = result.pose_landmarks[0]

    if len(landmarks) != 33:
        return None

    pose = np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks],
        dtype=np.float32,
    )

    return pose


def extract_pose33_from_video(video_path, model_path=None, sample_every=1):
    """영상에서 유효한 Pose 프레임을 순서대로 추출한다."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"video file does not exist: {video_path}")
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"cv2 failed to open video: {video_path}")

    poses = []
    skipped_frames = []
    frame_index = 0
    if hasattr(mp, "solutions"):
        detector = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_every == 0:
                    result = detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if result.pose_landmarks is None:
                        skipped_frames.append(frame_index)
                    else:
                        poses.append(np.array(
                            [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark],
                            dtype=np.float32,
                        ))
                frame_index += 1
        finally:
            capture.release()
            detector.close()
    else:
        detector = create_pose_landmarker(model_path)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame_index % sample_every == 0:
                    pose = extract_pose33_from_bgr_frame(frame, detector)
                    if pose is None:
                        skipped_frames.append(frame_index)
                    else:
                        poses.append(pose)
                frame_index += 1
        finally:
            capture.release()
            detector.close()

    if len(poses) < 30:
        raise ValueError(f"성공한 pose frame이 30개보다 적습니다: {len(poses)}")
    return np.stack(poses, axis=0).astype(np.float32), skipped_frames


def extract_pose33_from_image_folder(image_dir, model_path, max_frames=None):
    image_files = list_image_files(image_dir, recursive=False)

    if max_frames is not None:
        image_files = image_files[:int(max_frames)]

    detector = create_pose_landmarker(model_path)

    poses = []
    frame_files = []
    skipped_files = []

    try:
        for idx, image_path in enumerate(image_files):
            if idx % 200 == 0:
                print(f"MediaPipe Tasks 처리 중: {idx}/{len(image_files)}")

            try:
                pose = extract_pose33_from_image(image_path, detector=detector)
            except Exception as e:
                skipped_files.append({
                    "path": str(image_path),
                    "reason": str(e),
                })
                continue

            if pose is None:
                skipped_files.append({
                    "path": str(image_path),
                    "reason": "pose_not_found",
                })
                continue

            poses.append(pose)
            frame_files.append(str(image_path))

    finally:
        detector.close()

    if len(poses) < 30:
        raise ValueError(f"성공한 pose frame이 30개보다 적습니다: {len(poses)}")

    pose_array = np.stack(poses, axis=0).astype(np.float32)

    return pose_array, frame_files, skipped_files


def json_default(obj):
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    return str(obj)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
