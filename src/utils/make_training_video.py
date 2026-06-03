import argparse
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


STEP_RE = re.compile(r"step_(\d+)_(binary|prob)\.png$")


def parse_step(path: Path) -> int:
    match = STEP_RE.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected sample filename: {path.name}")
    return int(match.group(1))


def collect_frames(sample_dir: Path, kind: str, max_frames: int | None) -> list[Path]:
    frames = [
        path for path in sample_dir.glob(f"step_*_{kind}.png")
        if STEP_RE.match(path.name)
    ]
    frames = sorted(frames, key=parse_step)
    if max_frames is not None and len(frames) > max_frames:
        indices = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[int(index)] for index in indices]
    if not frames:
        raise FileNotFoundError(f"No step_*_{kind}.png files found in {sample_dir}")
    return frames


def load_font(size: int) -> ImageFont.ImageFont:
    for font_name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(font_name, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def add_label(image: Image.Image, text: str, font_size: int) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    pad = max(8, font_size // 2)
    x = pad
    y = pad
    draw.rectangle(
        (x - pad // 2, y - pad // 2, x + text_w + pad // 2, y + text_h + pad // 2),
        fill=(0, 0, 0, 150),
    )
    draw.text((x, y), text, fill=(255, 255, 255, 255), font=font)
    return image


def prepare_frame(
    path: Path,
    scale: float,
    annotate: bool,
    font_size: int,
) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    if scale != 1.0:
        width = max(1, int(round(image.width * scale)))
        height = max(1, int(round(image.height * scale)))
        image = image.resize((width, height), Image.Resampling.BICUBIC)
    if annotate:
        step = parse_step(path)
        image = add_label(image, f"step {step:,}", font_size)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def write_mp4(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    scale: float,
    annotate: bool,
    font_size: int,
    hold_last: int,
) -> None:
    first = prepare_frame(frame_paths[0], scale, annotate, font_size)
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    try:
        writer.write(first)
        last_frame = first
        for path in frame_paths[1:]:
            frame = prepare_frame(path, scale, annotate, font_size)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            last_frame = frame
        for _ in range(max(0, hold_last)):
            writer.write(last_frame)
    finally:
        writer.release()


def write_gif(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    scale: float,
    annotate: bool,
    font_size: int,
    hold_last: int,
) -> None:
    frames = []
    for path in frame_paths:
        bgr = prepare_frame(path, scale, annotate, font_size)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    if hold_last > 0:
        frames.extend([frames[-1]] * hold_last)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(round(1000 / fps))
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a video from saved GAN training sample grids.")
    parser.add_argument("--sample_dir", type=Path, default=Path("outputs/samples/lightning"))
    parser.add_argument("--kind", choices=("binary", "prob"), default="binary")
    parser.add_argument("--output", type=Path, default=Path("outputs/videos/wgan_training_binary.mp4"))
    parser.add_argument("--fps", type=float, default=6.0)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--hold_last", type=int, default=12)
    parser.add_argument("--font_size", type=int, default=32)
    parser.add_argument("--no_annotate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame_paths = collect_frames(args.sample_dir, args.kind, args.max_frames)
    output_path = args.output
    annotate = not args.no_annotate
    if output_path.suffix.lower() == ".gif":
        write_gif(
            frame_paths,
            output_path,
            args.fps,
            args.scale,
            annotate,
            args.font_size,
            args.hold_last,
        )
    else:
        write_mp4(
            frame_paths,
            output_path,
            args.fps,
            args.scale,
            annotate,
            args.font_size,
            args.hold_last,
        )
    print(f"Wrote {len(frame_paths)} frames to {output_path}")


if __name__ == "__main__":
    main()
