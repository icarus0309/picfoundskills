"""Compatibility entry point for the self-contained Skill repair helper."""
import runpy
from pathlib import Path

if __name__=="__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1]/"skills/image-case-retrieval/scripts/repair_screen_identifiers.py"),run_name="__main__")
