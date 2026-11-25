"""Project storage paths and atomic JSON persistence shared by backend services."""
import json
import uuid
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(path, value, *, indent=2):
    import time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        for attempt in range(3):
            try:
                temporary.replace(path)
                break
            except PermissionError:
                if attempt == 2:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    finally:
        temporary.unlink(missing_ok=True)
