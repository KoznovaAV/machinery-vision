"""Check current images, features and source-separated controls."""

import json
from src.data.manifest import audit

if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(not result["passed"])
