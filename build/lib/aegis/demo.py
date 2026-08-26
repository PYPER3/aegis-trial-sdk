"""Preview/report command for the public Trial SDK scaffold."""

import argparse
from pathlib import Path

from .report import write_evaluation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="AEGIS Trial SDK evaluation-report preview")
    parser.add_argument("--output", type=Path, default=Path("AEGIS_EVALUATION_RESULT.md"))
    parser.add_argument("--preview", action="store_true", help="Write clearly simulated report output.")
    args = parser.parse_args()
    if not args.preview:
        parser.error("This scaffold command supports --preview only. A separately supplied compatible Trial Core wheel is required for live detection.")
    report = write_evaluation_report(args.output, {"simulated": True})
    print(f"Simulated preview written to {report}")


if __name__ == "__main__":
    main()
