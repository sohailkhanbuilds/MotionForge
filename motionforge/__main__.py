"""Run MotionForge with a non-interactive Matplotlib backend."""
import os
# MediaPipe's legacy Solutions import matplotlib; Colab may inject a backend
# module unavailable in this isolated Python environment.
os.environ["MPLBACKEND"] = "Agg"
from motionforge.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
