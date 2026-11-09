"""Download pinned public releases only. No model downloads or private Drive reads."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(url):
    request = urllib.request.Request(url, headers={"User-Agent": "SPNER-research-data-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/raw/public_v1"))
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("Use a new versioned raw directory; do not overwrite raw data")
    hf_id = "ljvmiranda921/tlunified-ner"
    hf_revision = "1d43a4a60df51660aa8746e8952c536ba9483c72"
    ceb_revision = "1239289791b86f4694509f022c9b802dd17c9ed7"
    hil_revision = "cfd02cb0fdb2f1aadb75106c7a779c17db687ada"
    files = []
    for split in ("train", "validation", "test"):
        rel = f"data/{split}-00000-of-00001.parquet"
        files.append((f"tlunified_ner/{split}.parquet",
                      f"https://huggingface.co/datasets/{hf_id}/resolve/{hf_revision}/{rel}"))
    files.append(("tlunified_ner/README.md", f"https://huggingface.co/datasets/{hf_id}/resolve/{hf_revision}/README.md"))
    for filename in ("final-1.txt", "final-2.txt"):
        files.append((f"cebuaner/{filename}",
                      f"https://raw.githubusercontent.com/mebzmoren/CebuaNER/{ceb_revision}/data/annotated_data/{filename}"))
    files.append(("hiligayner/final_combined_all.jsonl",
                  f"https://raw.githubusercontent.com/jvlzloons/HiligayNER/{hil_revision}/Dataset/Annotated%20dataset/final_combined_all.jsonl"))
    for repo, name, revision in (("mebzmoren/CebuaNER", "cebuaner", ceb_revision),
                                 ("jvlzloons/HiligayNER", "hiligayner", hil_revision)):
        files.append((f"{name}/README.md", f"https://raw.githubusercontent.com/{repo}/{revision}/README.md"))
    manifest = {"downloaded_utc": datetime.now(timezone.utc).isoformat(),
                "revisions": {"tlunified_ner": hf_revision, "cebuaner": ceb_revision, "hiligayner": hil_revision},
                "files": [], "status": "raw_only_not_harmonized"}
    for relative, url in files:
        payload = fetch(url)
        path = args.out / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
        manifest["files"].append({"path": relative, "source_url": url,
                                  "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        print(f"Downloaded {relative}: {len(payload)} bytes", flush=True)
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
