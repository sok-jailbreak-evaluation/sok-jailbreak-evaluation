"""Freeze response-labelled datasets without consulting an evaluator or a tokenizer."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import fcntl
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
DATASETS = ("wildguardtest", "harmbench_response", "jailbreakbench_response")
HISTORICAL_DATASETS = ("xstest_response",)
DEFAULT_DATA = ROOT / "data_jbb600"
LABELS = ("harmful", "unharmful")
ALIASES = {"wildguard": "wildguardtest", "harmbench": "harmbench_response", "jailbreakbench": "jailbreakbench_response",
           "jbb": "jailbreakbench_response", "xstest": "xstest_response"}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    if not isinstance(value, (str, bytes)):
        value = canonical(value)
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def frozen_write(path, content):
    path = Path(path)
    data = content.encode("utf-8") if isinstance(content, str) else content
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Refusing to overwrite a different frozen artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temp.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def as_json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def as_jsonl(rows):
    return "".join(canonical(row) + "\n" for row in rows)


def majority(item):
    votes = [item.get(f"human_{i}") for i in range(3)]
    if any(type(v) is not int or v not in (0, 1) for v in votes):
        raise ValueError("Three explicit binary integer human votes are required")
    return int(sum(votes) >= 2)


def jbb_majority(item):
    """Validate the published human aggregate; never use the classifier columns."""
    def binary(value):
        if type(value) not in (str, int) or value not in ("0", "1", 0, 1):
            raise ValueError("Explicit binary human annotation required")
        return int(value)
    votes = [binary(item.get(f"human{i}")) for i in range(1, 4)]
    result = int(sum(votes) >= 2)
    if result != binary(item.get("human_majority")):
        raise ValueError("Published JBB human_majority disagrees with individual votes")
    return result


def normalize_datasets(names):
    names = list(DATASETS) if names == ["all"] else [ALIASES.get(n, n) for n in names]
    if not names or len(names) != len(set(names)) or any(n not in (*DATASETS, *HISTORICAL_DATASETS) for n in names):
        raise ValueError(f"Choose distinct dataset IDs from {DATASETS}")
    return names


def rng_for(seed, *parts):
    return random.Random(int(digest([seed, *parts]), 16))


def sample_balanced(rows, count, seed, dataset, allow_shortfall=False):
    """Max-min category allocation with finite capacities and no replacement."""
    chosen, allocations = {}, {}
    for label in LABELS:
        pools = defaultdict(list)
        for row in rows:
            if row["sampling_group"] == label:
                pools[tuple(row["sampling_stratum"])].append(row)
        available = sum(map(len, pools.values()))
        if available < 1 or (available < count and not allow_shortfall):
            raise ValueError(f"{dataset}/{label}: only {available} distinct labelled rows; requested {count}. No repetition or relabelling is allowed.")
        actual_count = min(count, available) if allow_shortfall else count
        categories = sorted(pools)
        rng_for(seed, dataset, label, "quota_order").shuffle(categories)
        for category in categories:
            pools[category].sort(key=lambda r: r["source_record_id"])
            rng_for(seed, dataset, label, "within_stratum", list(category)).shuffle(pools[category])
        quota = {category: 0 for category in categories}
        for _ in range(actual_count):
            eligible = [c for c in categories if quota[c] < len(pools[c])]
            category = min(eligible, key=lambda c: quota[c])
            quota[category] += 1
        selected = [r for category in categories for r in pools[category][:quota[category]]]
        rng_for(seed, dataset, label, "output_order").shuffle(selected)
        chosen[label] = selected
        allocations[label] = [{"stratum": list(c), "available": len(pools[c]), "selected": quota[c]} for c in sorted(categories)]
        if actual_count >= len(categories):
            assert all(q > 0 for q in quota.values())
    result = [row for pair in zip_longest(chosen["harmful"], chosen["unharmful"]) for row in pair if row is not None]
    assert len({r["sample_id"] for r in result}) == sum(map(len, chosen.values()))
    return result, allocations


def base_record(dataset, source_id, prompt, response, label, stratum, context=""):
    if not isinstance(prompt, str) or not prompt.strip() or not isinstance(response, str) or not response.strip():
        raise ValueError("A complete nonempty prompt and response are required")
    if label not in LABELS:
        raise ValueError("Unknown response sampling label")
    return {"schema_version": 1, "dataset_id": dataset, "sample_id": f"{dataset}:{source_id}",
            "source_record_id": str(source_id), "prompt": prompt, "goal": prompt, "response": response,
            "context": context, "sampling_group": label, "sampling_stratum": list(stratum),
            "initial_experiment_eligible": label == "unharmful",
            "initial_optimization_direction": "up" if label == "unharmful" else None,
            "input_pair_id": digest({"prompt": prompt, "response": response, "context": context})}


def load_sources(dataset, cfg):
    import pyarrow.parquet as pq
    rows, excluded, sources = [], [], []
    if dataset == "wildguardtest":
        source = (ROOT / cfg["wildguard_source"]).resolve()
        raw = pq.read_table(source).to_pylist()
        for index, item in enumerate(raw):
            if item["response_harm_label"] not in LABELS:
                excluded.append({"source_row_index": index, "reason": "missing_or_unknown_response_harm_label"})
                continue
            row = base_record(dataset, f"row-{index:04d}", item["prompt"], item["response"], item["response_harm_label"], [item["subcategory"] or "unknown"])
            row.update(source_row_index=index, source_dataset="allenai/wildguardmix", source_configuration="wildguardtest",
                       source_split="test", source_revision=None, label_scope="response_harmfulness", label_source="original_human_response_harm_label",
                       native_label=item["response_harm_label"], response_harm_label=item["response_harm_label"],
                       prompt_harm_label=item["prompt_harm_label"], subcategory=item["subcategory"], original_record=item)
            rows.append(row)
        sources.append({"path": str(source.relative_to(REPO)), "rows": len(raw),
                        "revision": None, "provenance": "existing local WildGuardTest parquet; upstream revision is unknown, not inferred"})
    elif dataset == "harmbench_response":
        folder = ROOT / "sources/harmbench"
        provenance = read(ROOT / "sources/harmbench_source_manifest.json")
        assert provenance["revision"] == cfg["harmbench_revision"]
        source = folder / "data/classifier_val_sets/text_behaviors_val_set.json"
        meta_path = folder / "data/behavior_datasets/harmbench_behaviors_text_all.csv"
        raw = read(source)
        with meta_path.open() as stream:
            metadata = {m["BehaviorID"]: m for m in csv.DictReader(stream)}
        for behavior_id, items in raw.items():
            for index, item in enumerate(items):
                native = majority(item)
                if behavior_id not in metadata:
                    excluded.append({"behavior_id": behavior_id, "item_index": index, "human_votes": [item[f"human_{i}"] for i in range(3)],
                                     "native_majority_label": native, "reason": "BehaviorID_absent_from_pinned_official_text_metadata"})
                    continue
                meta = metadata[behavior_id]
                assert item["behavior_id"] == behavior_id
                label = "harmful" if native else "unharmful"
                row = base_record(dataset, f"{behavior_id}:{index}", meta["Behavior"], item["generation"], label,
                                  [meta["SemanticCategory"], meta["FunctionalCategory"]], meta["ContextString"])
                row.update(behavior_id=behavior_id, item_index=index, test_case=item["test_case"], source_dataset="centerforaisafety/HarmBench", source_split="classifier_val_sets/text_behaviors_val_set",
                           source_revision=cfg["harmbench_revision"], semantic_category=meta["SemanticCategory"],
                           functional_category=meta["FunctionalCategory"], model=item["model"], method=item["method"],
                           label_scope="goal_conditioned_harmful_behavior_success", label_source="three_human_votes_majority",
                           native_label=native, response_harm_label=None, global_response_harmfulness="not_separately_annotated",
                           human_votes=[item[f"human_{i}"] for i in range(3)], behavior_metadata=meta, original_record=item)
                rows.append(row)
        sources = [{"path": str(p.relative_to(REPO)), } for p in [source, meta_path]]
    elif dataset == "jailbreakbench_response":
        provenance = read(ROOT / "sources/jailbreakbench_source_manifest.json")
        assert provenance["revision"] == cfg["jailbreakbench_revision"]
        source = ROOT / "sources/jailbreakbench/data/judge-comparison.csv"
        with source.open() as stream:
            raw = list(csv.DictReader(stream))
        for index, item in enumerate(raw):
            native = jbb_majority(item)
            label = "harmful" if native else "unharmful"
            # This official response subset has no Category column. The separate
            # behavior-only lists are not a complete taxonomy for these goals.
            row = base_record(dataset, f"row-{index:04d}", item["goal"], item["target_response"], label,
                              ["no_official_category"])
            row.update(source_row_index=index, original_index=item["Index"], test_case=item["prompt"],
                       source_dataset="JailbreakBench/JBB-Behaviors",
                       source_configuration="judge_comparison", source_split="test",
                       source_revision=cfg["jailbreakbench_revision"], native_label=native,
                       human_votes=[int(item[f"human{i}"]) for i in range(1, 4)],
                       native_human_majority=int(item["human_majority"]), label_source="three_human_votes_verified_published_majority",
                       label_scope="goal_conditioned_jailbreak_success", response_harm_label=None,
                       global_response_harmfulness="not_separately_annotated", category=None,
                       category_source="not_provided_in_judge_comparison", original_record=item)
            rows.append(row)
        sources = [{"path": str(source.relative_to(REPO)), "rows": len(raw), "revision": cfg["jailbreakbench_revision"],
                    "provenance": "official actual responses with three human votes; goal/prompt/target_response kept separately"}]
    elif dataset == "xstest_response":
        source = ROOT / "sources/xstest_response/data/response_harmfulness-00000-of-00001.parquet"
        if not source.exists():
            raise FileNotFoundError("XSTest-Response response_harmfulness parquet is not available; do not use refusal labels or prompt labels as replacements")
        provenance = read(ROOT / "sources/xstest_local_source_manifest.json")
        raw = pq.read_table(source).to_pylist()
        for index, item in enumerate(raw):
            if item["label"] not in LABELS:
                excluded.append({"source_row_index": index, "reason": "missing_or_unknown_response_harmfulness_label"})
                continue
            row = base_record(dataset, f"row-{index:04d}", item["prompt"], item["response"], item["label"], [item["prompt_harm_category"] or "unknown"])
            row.update(source_row_index=index, source_dataset="allenai/xstest-response", source_split="response_harmfulness",
                       source_revision=provenance["verified_upstream_revision"], expected_upstream_revision=cfg["xstest_revision"],
                       label_scope="response_harmfulness", label_source="original_response_harmfulness_split_label",
                       native_label=item["label"], response_harm_label=item["label"], prompt_type=item["prompt_type"],
                       prompt_harm_category=item["prompt_harm_category"], original_record=item)
            rows.append(row)
        sources = [{"path": str(source.relative_to(REPO)), "rows": len(raw),
                    "provenance": "user supplied local response_harmfulness parquet; exact upstream revision not authenticated"}]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    ids = [r["sample_id"] for r in rows]
    assert len(ids) == len(set(ids))
    duplicate_pairs = defaultdict(list)
    for row in rows:
        duplicate_pairs[row["input_pair_id"]].append(row["sample_id"])
    if any(len(v) > 1 for v in duplicate_pairs.values()):
        raise ValueError(f"Repeated exact evaluation inputs in {dataset}; review duplicate provenance before sampling")
    return rows, excluded, sources


def build(datasets, output, per_label=100, seed=42, allow_shortfall=False):
    cfg = read(ROOT / "config.json")
    selected, manifests = {}, {}
    # Validate every requested pool before creating any final outputs.
    for dataset in datasets:
        rows, excluded, sources = load_sources(dataset, cfg)
        samples, allocations = sample_balanced(rows, per_label, seed, dataset, allow_shortfall)
        selected[dataset] = samples
        manifests[dataset] = {"dataset_id": dataset, "eligible": dict(Counter(r["sampling_group"] for r in rows)),
                              "selected": dict(Counter(r["sampling_group"] for r in samples)), "sources": sources,
                              "exclusions": excluded, "strata": allocations,
                              "requested_per_label": per_label,
                              "shortfalls": {label: max(0, per_label - sum(r["sampling_group"] == label for r in samples)) for label in LABELS},
                              "selected_source_ids": [r["source_record_id"] for r in samples],
                              "sample_ids": [r["sample_id"] for r in samples]}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        content = {}
        for dataset, samples in selected.items():
            content[f"{dataset}/samples.jsonl"] = as_jsonl(samples)
            for label in LABELS:
                content[f"{dataset}/{label}.jsonl"] = as_jsonl([r for r in samples if r["sampling_group"] == label])
            content[f"{dataset}/manifest.json"] = as_json(manifests[dataset])
        all_samples = [r for name in datasets for r in selected[name]]
        content["combined.jsonl"] = as_jsonl(all_samples)
        content["unharmful_up.jsonl"] = as_jsonl([r for r in all_samples if r["sampling_group"] == "unharmful"])
        manifest = {"schema_version": 1, "datasets": datasets, "samples": len(all_samples), "seed": seed,
                    "per_dataset_per_label": per_label, "sampling_config": cfg, "groups": LABELS,
                    "allow_shortfall": allow_shortfall,
                    "requested_samples": len(datasets) * per_label * len(LABELS),
                    "shortfalls": {name: m["shortfalls"] for name, m in manifests.items()},
                    "sampling_requirements_met": all(not n for m in manifests.values() for n in m["shortfalls"].values()),
                    "selection_uses_model_scores": False, "replacement": False, "initial_direction": "up_only",
                    "source_code": [p.name for p in sorted(ROOT.glob("*.py"))],
                    "files": {name: {"bytes": len(text.encode())} for name, text in content.items()}}
        content["manifest.json"] = as_json(manifest)
        for name, text in content.items():
            frozen_write(output / name, text)
    print(as_json({"output": str(output), "samples": len(all_samples), "datasets": datasets,
                   "per_dataset_per_label": per_label, }), end="")
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["all"])
    p.add_argument("--output", type=Path, default=DEFAULT_DATA)
    p.add_argument("--per-label", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-shortfall", action="store_true", help="Keep all available rows for an undersized label and explicitly mark unmet requested quotas. Never duplicate rows.")
    args = p.parse_args()
    if args.per_label < 1:
        p.error("--per-label must be positive")
    build(normalize_datasets(args.datasets), args.output, args.per_label, args.seed, args.allow_shortfall)


if __name__ == "__main__":
    main()
