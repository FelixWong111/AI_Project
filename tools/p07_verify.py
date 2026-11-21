"""Execute reproducible smoke commands and retain their actual outputs."""
import json
import subprocess
import sys
import platform
from datetime import datetime, timezone
from pathlib import Path

commands = [
    ["-m", "unittest", "discover", "-s", "tests", "-v"],
    ["tools/p07_demo.py"],
    ["app.py", "--headless", "--image", "tmp/p07-demo/mock-input.png", "--max-frames", "1"],
    ["app.py", "--headless", "--video", "tmp/p07-demo/mock-input.avi", "--max-frames", "5"],
    ["app.py", "--video", "tmp/p07-demo/mock-input.avi", "--max-frames", "5"],
    ["app.py", "--headless", "--image", "tmp/p07-demo/mock-input.png", "--backend", "real"],
    ["app.py", "--headless", "--image", "tmp/p07-demo/absent.png"],
]
records = []
for command in commands:
    result = subprocess.run([sys.executable, *command], capture_output=True, text=True, timeout=40)
    expected = 1 if "real" in command or "tmp/p07-demo/absent.png" in command else 0
    records.append(dict(command=[sys.executable, *command], exit_code=result.returncode, expected_exit_code=expected,
                        passed=result.returncode == expected, stdout=result.stdout, stderr=result.stderr))
    print(command, "PASS" if records[-1]["passed"] else "FAIL")
path = Path("docs/p07/verification.json")
path.parent.mkdir(parents=True, exist_ok=True)
import cv2
import numpy as np
path.write_text(json.dumps(dict(timestamp=datetime.now(timezone.utc).isoformat(),
    environment=dict(python=platform.python_version(), platform=platform.platform(), opencv=cv2.__version__, numpy=np.__version__),
    records=records), ensure_ascii=False, indent=2), encoding="utf-8")
raise SystemExit(0 if all(r["passed"] for r in records) else 1)
