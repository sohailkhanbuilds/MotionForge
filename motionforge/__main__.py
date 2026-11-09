"""Run MotionForge commands with a non-interactive Matplotlib backend."""
import os
import sys

# MediaPipe's legacy Solutions import matplotlib; Colab may inject a backend
# module unavailable in this isolated Python environment.
os.environ["MPLBACKEND"] = "Agg"

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "project":
        from motionforge.project_cli import main
        raise SystemExit(main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "blender-script":
        from motionforge.blender_cli import main
        raise SystemExit(main(sys.argv[2:]))

    from motionforge.cli import main
    raise SystemExit(main())
