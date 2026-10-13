"""
Add the top-up TriviaQA samples to the independent samples built by 124.

    python3 126_add_topup_samples.py <main_output_dir> <topup_output_dir>

The first Kaggle run (validation split) yielded 14 independent samples after
deduplication. The top-up kernel drew 11 x 700 further questions from the
TRAIN split, deduplicated in-kernel and excluding every validation question.
This script refuses to add them unless that holds on the downloaded files:

  - every top-up question id and normalised question text is unique;
  - none appears in the 14 existing samples (by id or by text);
  - each shard is exactly 700 x 32 x 4096;
  - labels are re-derived from the saved answers with 124's rule.

It then writes audit_input_14.npz ... audit_input_24.npz, extends
independent_samples.json, and updates the DEDUPED count to 25.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("build", ROOT / "124_build_independent_samples.py")
build = importlib.util.module_from_spec(spec); spec.loader.exec_module(build)


def norm(q):
    return " ".join(q.lower().split())


def main():
    M, T = Path(sys.argv[1]), Path(sys.argv[2])
    meta = json.load(open(M / "independent_samples.json"))
    n0 = meta["n_samples"]
    assert n0 == 14 and int((M / "DEDUPED").read_text()) == 14, "expected the 14-sample state from 124"
    old_ids = {q for s in meta["samples"] for q in s["qids"]}
    old_txt = set()
    for k in range(25):
        for j in json.load(open(M / f"shard_{k:02d}_answers.json")):
            old_txt.add(norm(j["question"]))      # every validation question seen, used or not
    shards = sorted(T.glob("shard_[0-9][0-9].npz"))
    assert len(shards) == 11, f"expected 11 top-up shards, found {len(shards)}"
    new_ids, new_txt = set(), set()
    added = []
    for k, sp in enumerate(shards):
        J = json.load(open(T / f"shard_{k:02d}_answers.json"))
        assert len(J) == build.SAMPLE
        for j in J:
            t = norm(j["question"])
            assert j["qid"] not in old_ids and t not in old_txt, f"top-up question overlaps the first run: {j['qid']}"
            assert j["qid"] not in new_ids and t not in new_txt, f"duplicate inside the top-up: {j['qid']}"
            new_ids.add(j["qid"]); new_txt.add(t)
        H = np.load(sp)["hidden_states"]
        assert H.shape == (build.SAMPLE, 32, 4096), H.shape
        y = np.array([0 if build.correct(j["answer"], j["aliases"]) else 1 for j in J], dtype=np.int64)
        s = n0 + k
        np.savez(M / f"audit_input_{s:02d}.npz", hidden_states=H, labels=y)
        added.append({"sample": s, "source": "train split (top-up)",
                      "hallucination_rate": float(y.mean()), "qids": [j["qid"] for j in J]})
        print(f"sample {s}: hall_rate {y.mean():.3f}")
    meta["samples"] += added
    meta["n_samples"] = n0 + len(added)
    meta["topup"] = {"n_samples": len(added), "split": "train", "disjoint_from_validation": True}
    with open(M / "independent_samples.json", "w") as f:
        json.dump(meta, f)
    (M / "DEDUPED").write_text(f"{meta['n_samples']}\n")
    ids = [q for s in meta["samples"] for q in s["qids"]]
    print(f"total independent samples {meta['n_samples']}; question ids {len(ids)}, all distinct: {len(ids) == len(set(ids))}")


if __name__ == "__main__":
    main()
