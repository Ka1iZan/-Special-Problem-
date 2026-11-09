from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new manifest path; existing manifests are immutable")
    if not args.root.is_dir():
        raise NotADirectoryError(args.root)
    files = []
    for p in sorted(x for x in args.root.rglob("*") if x.is_file()):
        files.append({"path": str(p.relative_to(args.root)), "bytes": p.stat().st_size, "sha256": sha256(p)})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump({"root": str(args.root), "files": files}, stream, indent=2)

if __name__ == "__main__":
    main()
