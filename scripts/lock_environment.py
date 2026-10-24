"""Write a deterministic environment lock for the feature-geometry study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_inventory() -> dict[str, str]:
    files: dict[str, str] = {}
    for base in (ROOT / "src", ROOT / "scripts", ROOT / "tests"):
        for path in sorted(base.rglob("*.py")):
            files[str(path.relative_to(ROOT)).replace("\\", "/")] = sha256_file(path)
    for path in (ROOT / "MANIFEST.md", ROOT / "README.md"):
        files[path.name] = sha256_file(path)
    return files


def git_metadata() -> dict[str, object]:
    repo = ROOT / ".." / "overleaf"
    if not (repo / ".git").exists():
        return {"repo": None, "commit": None, "dirty": None}
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return {"repo": str(repo), "commit": commit, "dirty": bool(status.stdout.strip())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "env")
    args = parser.parse_args()
    packages = {dist.metadata["Name"]: dist.version for dist in importlib.metadata.distributions() if dist.metadata.get("Name")}
    lock = {
        "schema_version": "feature_geometry.environment_lock.v1",
        "python": {"version": platform.python_version(), "implementation": platform.python_implementation(), "executable": sys.executable},
        "platform": {"system": platform.system(), "release": platform.release(), "machine": platform.machine(), "processor": platform.processor()},
        "packages": dict(sorted(packages.items(), key=lambda item: item[0].lower())),
        "study_repo": git_metadata(),
        "source_inventory": source_inventory(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "environment.lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    requirements = "\n".join(f"{name}=={version}" for name, version in sorted(packages.items(), key=lambda item: item[0].lower())) + "\n"
    (args.output_dir / "requirements.freeze.txt").write_text(requirements, encoding="utf-8")
    print(json.dumps({"packages": len(packages), "source_files": len(lock["source_inventory"]), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
