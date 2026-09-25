#!/usr/bin/env python3
"""Assemble quadtree-lensed frames into an animated GIF.

Keeps every source PNG untouched under img/quadtree_lensed_frames/;
outputs img/quadtree_lensed_fit.gif.  Coarse frames hold longer, and
the most refined frame (level 8) holds longest as a capstone.
"""
import glob, os
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
frames_dir = os.path.join(root, "img", "quadtree_lensed_frames")
out = os.path.join(root, "img", "quadtree_lensed_fit.gif")

paths = sorted(glob.glob(os.path.join(frames_dir, "quadtree_lensed_P1_lv*.png")))
assert len(paths) == 8, f"expected 8 frames, got {len(paths)}"

# per-frame hold in ms (index 0 = level 1)
durations = [700, 650, 600, 550, 500, 500, 450, 1300]
imgs = [Image.open(p) for p in paths]

imgs[0].save(out, save_all=True, append_images=imgs[1:],
             duration=durations, loop=0, optimize=True)
print(f"wrote {out}  ({len(paths)} frames, {os.path.getsize(out)//1024} KB)")
