"""Segmentation, reference-mask tracking, and measurements for image sequences."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TypeAlias

import numpy as np
import pandas as pd
from skimage import color, exposure, io, transform, util
from skimage.restoration import denoise_bilateral
from skimage.segmentation import find_boundaries
from ultralytics import YOLO

MAX_SIDE = 680
CONFIDENCE = 0.25
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"})

ImageArray: TypeAlias = np.ndarray


def natural_sort_key(path: Path) -> tuple[tuple[bool, int | str], ...]:
    """Sort numbered frame names in acquisition order."""
    return tuple(
        (part.isdigit(), int(part) if part.isdigit() else part.casefold())
        for part in re.split(r"(\d+)", path.name)
    )


def discover_frames(image_sequence: Path) -> list[Path]:
    """Return supported image files in natural filename order."""
    if not image_sequence.is_dir():
        raise NotADirectoryError(f"Image sequence directory not found: {image_sequence}")

    frame_paths = sorted(
        (
            path
            for path in image_sequence.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    if not frame_paths:
        raise FileNotFoundError(f"No supported image files found in {image_sequence}")
    return frame_paths


def preprocess_frame(image_path: Path, max_side: int = MAX_SIDE) -> np.ndarray:
    """Denoise, enhance contrast, and resize an image while preserving aspect ratio."""
    if max_side < 1:
        raise ValueError("max_side must be a positive integer")

    image = io.imread(image_path)
    if image.ndim == 2:
        image = color.gray2rgb(image)
    elif image.ndim == 3 and image.shape[-1] == 4:
        image = color.rgba2rgb(image)
    elif image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Unsupported image shape {image.shape}: {image_path}")

    image = util.img_as_float32(image)
    denoised = denoise_bilateral(
        image,
        win_size=9,
        sigma_color=75 / 255,
        sigma_spatial=75,
        channel_axis=-1,
    )
    lab = color.rgb2lab(denoised)
    height, width = lab.shape[:2]
    lab[:, :, 0] = exposure.equalize_adapthist(
        lab[:, :, 0] / 100,
        kernel_size=(max(1, height // 8), max(1, width // 8)),
        clip_limit=0.01,
    ) * 100
    enhanced = color.lab2rgb(lab)

    scale = max_side / max(height, width)
    output_shape = (
        max(1, round(height * scale)),
        max(1, round(width * scale)),
        3,
    )
    resized = transform.resize(
        enhanced,
        output_shape,
        anti_aliasing=scale < 1,
        preserve_range=True,
    )
    return util.img_as_ubyte(resized)


def segment_frame(
    model: YOLO,
    image: ImageArray,
    max_side: int = MAX_SIDE,
    confidence: float = CONFIDENCE,
) -> tuple[ImageArray, ImageArray]:
    """Return instance masks and confidence scores for one preprocessed frame."""
    result = model.predict(
        source=np.ascontiguousarray(image[..., ::-1]),
        imgsz=max_side,
        conf=confidence,
        retina_masks=True,
        verbose=False,
    )[0]

    if result.masks is None:
        empty_masks = np.empty((0, image.shape[0], image.shape[1]), dtype=bool)
        return empty_masks, np.empty(0, dtype=float)

    masks = result.masks.data.detach().cpu().numpy().astype(bool)
    if masks.shape[1:] != image.shape[:2]:
        raise ValueError("Prediction masks do not match the preprocessed frame dimensions.")
    if result.boxes is None:
        raise ValueError("Segmentation masks were returned without detection boxes.")
    confidences = result.boxes.conf.detach().cpu().numpy()
    return masks, confidences


def mask_iou_matrix(reference_masks: ImageArray, current_masks: ImageArray) -> ImageArray:
    """Compute current-mask-by-reference-mask intersection-over-union scores."""
    iou = np.zeros((len(current_masks), len(reference_masks)), dtype=float)
    for current_index, current_mask in enumerate(current_masks):
        for reference_index, reference_mask in enumerate(reference_masks):
            intersection = np.count_nonzero(current_mask & reference_mask)
            union = np.count_nonzero(current_mask | reference_mask)
            iou[current_index, reference_index] = intersection / union if union else 0.0
    return iou


def match_to_reference(reference_masks: ImageArray, current_masks: ImageArray) -> ImageArray:
    """Assign each current mask to its best unique first-frame reference, or -1."""
    current_track_ids = np.full(len(current_masks), -1, dtype=int)
    if len(reference_masks) == 0 or len(current_masks) == 0:
        return current_track_ids

    iou = mask_iou_matrix(reference_masks, current_masks)
    best_current = np.argmax(iou, axis=0)
    best_iou = iou[best_current, np.arange(len(reference_masks))]

    for current_index in np.unique(best_current):
        candidate_references = np.flatnonzero(best_current == current_index)
        selected_reference = candidate_references[np.argmax(best_iou[candidate_references])]
        current_track_ids[current_index] = selected_reference

    return current_track_ids


def draw_tracks(
    image: ImageArray,
    masks: ImageArray,
    track_ids: ImageArray,
) -> ImageArray:
    """Tint tracked masks in RGB using a stable color for each track."""
    overlay = image.copy()
    for mask, track_id in zip(masks, track_ids):
        if track_id < 0:
            continue
        hue = (int(track_id) * 37 % 180) / 180
        track_color = (color.hsv2rgb(np.array([hue, 0.86, 1.0])) * 255).astype(np.uint8)
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * track_color).astype(np.uint8)
        overlay[find_boundaries(mask, mode="outer")] = track_color
    return overlay


def track_sequence(
    model_path: Path,
    image_sequence: Path,
    output_csv: Path = Path("tracker_measurements.csv"),
    max_side: int = MAX_SIDE,
    confidence: float = CONFIDENCE,
) -> tuple[pd.DataFrame, dict[int, ImageArray]]:
    """Run segmentation and tracking, save measurements, and return preview overlays."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    frame_paths = discover_frames(image_sequence)
    model = YOLO(str(model_path))

    records: list[dict[str, int | float | str]] = []
    preview_overlays: dict[int, ImageArray] = {}
    preview_indices = {0, len(frame_paths) // 2, len(frame_paths) - 1}
    reference_masks: ImageArray | None = None
    reference_shape: tuple[int, int] | None = None

    for frame_index, image_path in enumerate(frame_paths):
        frame = preprocess_frame(image_path, max_side=max_side)
        masks, confidences = segment_frame(
            model, frame, max_side=max_side, confidence=confidence
        )

        if frame_index == 0:
            reference_masks = masks.copy()
            reference_shape = frame.shape[:2]
            if len(reference_masks) == 0:
                raise ValueError(
                    "No spheroids were detected in the first frame; "
                    "reference tracks cannot be initialized."
                )
            track_ids = np.arange(len(masks), dtype=int)
        else:
            if frame.shape[:2] != reference_shape:
                raise ValueError(
                    "Frames differ in size after preprocessing; pixel-mask IoU "
                    "requires a common image grid."
                )
            track_ids = match_to_reference(reference_masks, masks)

        for detection_index, (mask, track_id) in enumerate(zip(masks, track_ids)):
            if track_id < 0:
                continue
            area_px2 = int(mask.sum())
            diameter_px = 2 * np.sqrt(area_px2 / np.pi)
            volume_px3 = (np.pi / 6) * diameter_px**3
            records.append(
                {
                    "frame_index": frame_index,
                    "frame_name": image_path.name,
                    "track_id": int(track_id),
                    "confidence": float(confidences[detection_index]),
                    "area_px2": area_px2,
                    "equivalent_diameter_px": diameter_px,
                    "sphere_equivalent_volume_px3": volume_px3,
                }
            )

        if frame_index in preview_indices:
            preview_overlays[frame_index] = draw_tracks(frame, masks, track_ids)

    measurements = pd.DataFrame.from_records(records)
    if measurements.empty:
        raise ValueError(
            "No tracked spheroid measurements were produced. "
            "Check the model, confidence, and first frame."
        )

    measurements.to_csv(output_csv, index=False)
    return measurements, preview_overlays