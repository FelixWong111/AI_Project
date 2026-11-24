"""Course-prescribed batch entrypoint."""
from src.inference.batch import main, load_cases, validate_counts

if __name__ == "__main__":
    raise SystemExit(main())
