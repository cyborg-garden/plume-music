"""Size of a built folder: total, file count, and the largest things in it."""
import json
import sys
from pathlib import Path


def measure(root: Path) -> dict:
    files = [p for p in root.rglob("*") if p.is_file() and not p.is_symlink()]
    sizes = {p: p.stat().st_size for p in files}
    groups: dict[str, int] = {}
    for p, size in sizes.items():
        rel = p.relative_to(root).parts
        # Under the folder that holds the libraries, one level down names the package.
        key = "/".join(rel[: min(len(rel), 2 if rel[0] != "Contents" else 4)])
        groups[key] = groups.get(key, 0) + size
    top = sorted(groups.items(), key=lambda kv: -kv[1])[:20]
    return {"path": str(root), "bytes": sum(sizes.values()), "mb": round(sum(sizes.values()) / 1e6, 1),
            "files": len(files), "largest": [[k, round(v / 1e6, 1)] for k, v in top]}


if __name__ == "__main__":
    doc = measure(Path(sys.argv[1]))
    if len(sys.argv) > 2:
        Path(sys.argv[2]).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(doc, indent=1))
