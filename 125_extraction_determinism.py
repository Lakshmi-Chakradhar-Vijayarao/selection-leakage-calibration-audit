"""
How reproducible is the batched extraction?  [supports 124]

TriviaQA rc.nocontext repeats questions, so the Kaggle run generated many
questions twice, in different batches. Each such pair is a free reproducibility
test of batched fp16 generation and state extraction: same prompt, same model,
same greedy decoding, different batch neighbours.

    python3 125_extraction_determinism.py <kaggle_output_dir> [n_pairs]
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    D = Path(sys.argv[1]); n_pairs = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    occ = {}
    for k in range(25):
        for i, j in enumerate(json.load(open(D / f"shard_{k:02d}_answers.json"))):
            occ.setdefault(j["qid"], []).append((k, i, j["answer"], j["label"]))
    pairs = [v[:2] for v in occ.values() if len(v) > 1][:n_pairs]
    same_ans = [a[2] == b[2] for a, b in pairs]
    same_lab = [a[3] == b[3] for a, b in pairs]
    need = {}
    for a, b in pairs:
        need.setdefault(a[0], set()).add(a[1]); need.setdefault(b[0], set()).add(b[1])
    st = {}
    for k in sorted(need):
        H = np.load(D / f"shard_{k:02d}.npz")["hidden_states"]
        for i in need[k]:
            st[(k, i)] = H[i].astype(np.float32)
    rel_same, rel_diff = [], []
    for (a, b), s in zip(pairs, same_ans):
        x, y = st[a[:2]], st[b[:2]]
        rel = float((np.linalg.norm(x - y, axis=1) / np.maximum(np.linalg.norm(x, axis=1), 1e-6)).max())
        (rel_same if s else rel_diff).append(rel)
    out = {"pairs": len(pairs), "identical_answer_share": float(np.mean(same_ans)),
           "same_label_share": float(np.mean(same_lab)),
           "same_answer_max_relative_state_diff": {"median": float(np.median(rel_same)), "max": float(np.max(rel_same))},
           "different_answer_n": len(rel_diff)}
    print(json.dumps(out, indent=1))
    with open(ROOT / "results" / "triviaqa_extraction_determinism.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
