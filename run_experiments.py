from pathlib import Path

from src.experiments import run_all


if __name__ == "__main__":
    run_all(Path(__file__).resolve().parent)
