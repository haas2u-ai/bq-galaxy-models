"""Shared paths for the video scripts.

Importing this module makes the scripts folder the working directory, so data files
are found wherever the script is started from. Videos are written to ../videos.
Set the environment variable BQ_MAX_FRAMES to render only the first N frames
(a quick test that the script runs), e.g.  BQ_MAX_FRAMES=60 python animate.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
VIDEOS = os.path.normpath(os.path.join(HERE, "..", "videos"))
CACHE = os.path.join(HERE, "cache")
os.makedirs(VIDEOS, exist_ok=True); os.makedirs(CACHE, exist_ok=True)
MAX_FRAMES = int(os.environ.get("BQ_MAX_FRAMES", "1000000000"))
def video(name): return os.path.join(VIDEOS, name)
def cache(name): return os.path.join(CACHE, name)
