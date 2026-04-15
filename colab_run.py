"""
Headless simulation for Google Colab (no pygame window).
Plots population time series with matplotlib (plt.show()).
Optional: save sampled frames as an animated GIF after the run.
"""
import argparse
import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib.pyplot as plt
import pygame
from PIL import Image

from Settings import HEIGHT, WIDTH, WITHIN_GROUP_DEMO
from Universe import Universe


def _maybe_display_in_notebook(gif_path: str) -> None:
    try:
        from IPython import get_ipython
        from IPython.display import Image as IPImage, display

        if get_ipython() is not None and os.path.isfile(gif_path):
            display(IPImage(filename=gif_path))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Run blob sim without display.")
    parser.add_argument("--steps", type=int, default=2000, help="Simulation frames")
    parser.add_argument(
        "--gif",
        metavar="PATH",
        default="",
        help="If set, append sampled frames to an animated GIF at this path",
    )
    parser.add_argument(
        "--gif-every",
        type=int,
        default=20,
        help="Record one GIF frame every N simulation frames",
    )
    parser.add_argument(
        "--gif-max-frames",
        type=int,
        default=72,
        help="Cap GIF length to keep file size small",
    )
    parser.add_argument(
        "--gif-duration-ms",
        type=int,
        default=75,
        help="Milliseconds per GIF frame",
    )
    args = parser.parse_args()

    pygame.init()
    universe = Universe()
    surf = pygame.Surface((WIDTH, HEIGHT))

    total_h = []
    selfish_h = []
    altru_h = []
    k_h = []
    gif_frames: list[Image.Image] = []

    for frame_idx in range(args.steps):
        universe.move()
        universe.update(surf)
        total_h.append(universe.total_population())
        if WITHIN_GROUP_DEMO:
            s, a = universe.count_behavior()
            selfish_h.append(s)
            altru_h.append(a)
            k_h.append(universe.carrying_capacity)

        if (
            args.gif
            and frame_idx % args.gif_every == 0
            and len(gif_frames) < args.gif_max_frames
        ):
            raw = pygame.image.tobytes(surf, "RGB")
            gif_frames.append(Image.frombytes("RGB", surf.get_size(), raw))

    if args.gif and gif_frames:
        out = os.path.abspath(args.gif)
        first, *rest = gif_frames
        first.save(
            out,
            save_all=True,
            append_images=rest,
            duration=args.gif_duration_ms,
            loop=0,
        )
        print(
            f"GIF saved: {out} ({len(gif_frames)} frames, "
            f"every {args.gif_every} sim frames)",
            file=sys.stderr,
        )
        _maybe_display_in_notebook(out)

    frames = list(range(len(total_h)))

    if WITHIN_GROUP_DEMO and selfish_h:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(frames, total_h, color="#2c5aa0", linewidth=1.2)
        axes[0].set_title("Total blobs")
        axes[0].set_xlabel("Frame")
        axes[0].set_ylabel("Count")
        axes[0].grid(alpha=0.3)

        axes[1].plot(frames, selfish_h, color="#e65548", linewidth=1, label="selfish")
        axes[1].plot(frames, altru_h, color="#559be8", linewidth=1, label="altruist")
        axes[1].plot(
            frames,
            k_h,
            color="#888888",
            linestyle="--",
            linewidth=0.9,
            label="K (ceiling)",
        )
        axes[1].set_title("Types + carrying capacity")
        axes[1].set_xlabel("Frame")
        axes[1].legend()
        axes[1].grid(alpha=0.3)
    else:
        plt.figure(figsize=(10, 4))
        plt.plot(frames, total_h, color="#2c5aa0", linewidth=1.2)
        plt.xlabel("Frame")
        plt.ylabel("Total blobs")
        plt.title("Total population over time")
        plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
