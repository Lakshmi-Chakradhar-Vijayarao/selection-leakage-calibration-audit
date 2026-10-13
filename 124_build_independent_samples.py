"""
Build the independent samples from the Kaggle TriviaQA run, without duplicates.

    python3 124_build_independent_samples.py <kaggle_output_dir>

WHY THIS STEP EXISTS. The kernel drew 17,500 rows from TriviaQA rc.nocontext
validation and cut them into 25 shards of 700. That split lists many questions
more than once -- once per evidence document in the reading-comprehension
release -- so the shards share questions. A question in two shards breaks the
independence the population analysis rests on, and a question twice in one
shard can sit on both sides of a selection/held-out split. Found by checking
the downloaded answers before any analysis was run on them (14 shards checked:
9,800 rows, 7,397 unique question ids).

WHAT IT DOES.
  1. Reads every shard's answers in the kernel's shuffled order and keeps the
     first occurrence of each question, by question id and by normalised text.
  2. Checks that each dropped duplicate's hidden states match the kept copy
     (greedy decoding on an identical prompt should reproduce them), so that
     dropping it loses nothing.
  3. Cuts the unique questions, still in shuffled order, into as many disjoint
     700-item samples as they fill.
  4. Re-derives every label from the saved answer with the corrected rule
     (alias containment across both punctuation normalisations), so no label
     depends on which kernel version wrote it.

Writes <dir>/audit_input_XX.npz for 122, <dir>/independent_samples.json (the
question ids in each sample, the duplicate check, the label rates) and the
marker <dir>/DEDUPED.
"""
import json
import re
import string
import sys
from pathlib import Path

import numpy as np

SAMPLE = int(__import__("os").environ.get("P2_SAMPLE_SIZE", 700))


def variants(s):
    s = s.lower()
    p = set(string.punctuation)
    out = set()
    for v in ("".join(c for c in s if c not in p), "".join(" " if c in p else c for c in s)):
        v = re.sub(r"\b(a|an|the)\b", " ", v)
        out.add(" ".join(v.split()))
    return out


def correct(answer, aliases):
    A = [f" {v} " for v in variants(answer)]
    L = set().union(*[variants(x) for x in aliases if x]) if aliases else set()
    return any(f" {l} " in a for l in L if l for a in A)


def main():
    D = Path(sys.argv[1])
    shards = sorted(D.glob("shard_[0-9][0-9].npz"))
    assert len(shards) == 25, f"expected 25 shards, found {len(shards)}"

    # pass 1: decide what to keep
    rows, seen_id, seen_text, dup_of = [], {}, {}, []
    for k, sp in enumerate(shards):
        J = json.load(open(D / f"shard_{k:02d}_answers.json"))
        for i, j in enumerate(J):
            key_t = " ".join(j["question"].lower().split())
            first = seen_id.get(j["qid"], seen_text.get(key_t))
            if first is None:
                seen_id[j["qid"]] = seen_text[key_t] = len(rows)
                rows.append({"shard": k, "row": i, "qid": j["qid"], "answer": j["answer"],
                             "aliases": j["aliases"],
                             "label": 0 if correct(j["answer"], j["aliases"]) else 1})
            else:
                dup_of.append(((k, i), first))
    n_rows = sum(len(json.load(open(D / f"shard_{k:02d}_answers.json"))) for k in range(25))
    n_samples = len(rows) // SAMPLE
    print(f"rows {n_rows} | unique questions {len(rows)} | duplicates dropped {len(dup_of)} "
          f"| independent samples of {SAMPLE}: {n_samples}")

    # pass 2: stream. Kept rows appear in shard order, so each output sample is
    # filled from consecutive source shards; only the current source shard and
    # one output buffer are in memory. Duplicates are checked against their
    # kept copy for a fixed subset of 300 pairs, enough to confirm that greedy
    # decoding reproduced the states.
    check = {}
    for (k, i), first in dup_of[:300]:
        check.setdefault(rows[first]["shard"], set()).add(rows[first]["row"])
    ref_states, diffs = {}, []
    dup_in = {}
    for (k, i), first in dup_of[:300]:
        dup_in.setdefault(k, []).append((i, first))

    meta = {"n_rows": n_rows, "n_unique": len(rows), "n_duplicates_dropped": len(dup_of),
            "sample_size": SAMPLE, "n_samples": n_samples, "samples": []}
    buf_H, buf_rows, s_out = [], [], 0
    kept_by_shard = {}
    for idx, r in enumerate(rows):
        kept_by_shard.setdefault(r["shard"], []).append(idx)
    for k in range(25):
        H = np.load(shards[k])["hidden_states"]
        for idx in kept_by_shard.get(k, []):
            r = rows[idx]
            if r["row"] in check.get(k, ()):
                ref_states[idx] = H[r["row"]].astype(np.float32)
            if s_out < n_samples:
                buf_H.append(H[r["row"]]); buf_rows.append(r)
                if len(buf_rows) == SAMPLE:
                    y = np.array([x["label"] for x in buf_rows], dtype=np.int64)
                    np.savez(D / f"audit_input_{s_out:02d}.npz", hidden_states=np.stack(buf_H), labels=y)
                    meta["samples"].append({"sample": s_out, "hallucination_rate": float(y.mean()),
                                            "qids": [x["qid"] for x in buf_rows]})
                    buf_H, buf_rows, s_out = [], [], s_out + 1
        for i, first in dup_in.get(k, []):
            diffs.append(float(np.abs(H[i].astype(np.float32) - ref_states[first]).max()))
        del H
    meta["duplicate_pairs_checked"] = len(diffs)
    meta["max_duplicate_state_difference"] = max(diffs) if diffs else None
    print(f"duplicate pairs checked {len(diffs)}; largest hidden-state difference "
          f"between a duplicate and its kept copy: {meta['max_duplicate_state_difference']:.4g}")

    all_ids = [q for smp in meta["samples"] for q in smp["qids"]]
    assert len(meta["samples"]) == n_samples
    assert len(all_ids) == len(set(all_ids)), "a question id appears in two samples"
    rates = [smp["hallucination_rate"] for smp in meta["samples"]]
    print(f"hallucination rate per sample {min(rates):.3f}-{max(rates):.3f}; "
          f"no question id in two samples: True")
    with open(D / "independent_samples.json", "w") as f:
        json.dump(meta, f)
    (D / "DEDUPED").write_text(f"{n_samples}\n")


if __name__ == "__main__":
    main()
