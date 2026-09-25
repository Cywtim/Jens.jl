#!/usr/bin/env python3
"""Assemble quadtree fit frames into an animated GIF.

Keeps every source PNG untouched under img/quadtree_fit_frames/;
outputs img/quadtree_fit.gif.  Frame durations scale with leaf growth:
coarse frames hold a bit longer, fine frames shorter, and the final
(most refined) frame holds longest as a capstone.
"""
import glob, os
from PIL import Image

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.dirname(here)
frames_dir = os.path.join(root, "img", "quadtree_fit_frames")
out = os.path.join(root, "img", "quadtree_fit.gif")

paths = sorted(glob.glob(os.path.join(frames_dir, "quadtree_P1_lv??.png")))
assert len(paths) == 10, f"expected 10 frames, got {len(paths)}"

# per-frame hold in ms (index 0 = level 1)
durations = [700, 600, 550, 500, 450, 450, 400, 400, 350, 1200]
imgs = [Image.open(p) for p in paths]

imgs[0].save(out, save_all=True, append_images=imgs[1:],
             duration=durations, loop=0, optimize=True)
print(f"wrote {out}  ({len(paths)} frames, {os.path.getsize(out)//1024} KB)")
