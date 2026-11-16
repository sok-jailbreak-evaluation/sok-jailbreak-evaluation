"""Select named frozen datasets and labels; never silently omit a requested dataset."""
import argparse
from pathlib import Path
from build import ROOT, DEFAULT_DATA, LABELS, normalize_datasets, read, digest, as_jsonl, frozen_write


def select(folder, datasets, group="all"):
    manifest = read(folder / "manifest.json")
    rows = []
    for dataset in datasets:
        if dataset not in manifest["datasets"]:
            raise ValueError(f"Requested dataset {dataset} is not frozen in {folder}; no implicit substitution")
        rel = f"{dataset}/samples.jsonl"
        raw = (folder / rel).read_bytes()
        assert digest(raw) == manifest["files"][rel]["sha256"]
        import json
        # JSON strings may contain U+0085/U+2028. Only LF delimits JSONL records.
        for line in raw.decode().split("\n"):
            if not line:
                continue
            row = json.loads(line)
            assert row["dataset_id"] == dataset
            if group == "all" or row["sampling_group"] == group:
                rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--group", choices=["all", *LABELS], default="unharmful")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = select(args.data, normalize_datasets(args.datasets), args.group)
    frozen_write(args.output, as_jsonl(rows))
    print(f"Selected {len(rows)} rows into {args.output}; dataset_id preserved")


if __name__ == "__main__":
    main()
