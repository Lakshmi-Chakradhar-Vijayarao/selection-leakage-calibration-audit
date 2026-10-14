"""
P2 -- population-level replication on independent samples.

WHY. Every interval in the paper comes from 50 re-splits of ONE 700-item
sample. Re-splits share data, so tests over them are split-conditional; the
standard re-split correction (Nadeau & Bengio, 2003) shows that on one sample
of this size neither the reuse cost nor the selection test is established at
population level. The clean design is many INDEPENDENT samples: run the full
audit inside each, and get population-level intervals from the spread across
samples, with no correction needed.

WHAT. 17,500 TriviaQA questions (rc.nocontext validation, shuffled with a fixed
seed) cut into 25 disjoint shards of 700 -- the main harness's sample size.
For every question: generate an answer with Mistral-7B-Instruct-v0.2, label
it, and save last-token hidden states at all 32 decoder layers.

FIDELITY TO THE MAIN HARNESS (GUARDIAN exp01, the source of the paper's
Mistral-7B TruthfulQA cache):
  model      mistralai/Mistral-7B-Instruct-v0.2, fp16, device_map="auto"
             (revision pinned below)
  prompt     the same template string, tokenised the same way
  decoding   greedy (do_sample=False)
  states     forward hooks on model.model.layers, i.e. each decoder layer's
             output before the final norm, at the last token of prompt +
             answer, with a trailing EOS stripped first ("Flaw 9 fix")
Departures, stated:
  dataset    TriviaQA instead of TruthfulQA -- TruthfulQA has 817 questions,
             too few for 25 independent samples of 700
  length     max_new_tokens 32 (TriviaQA answers are short entities) vs 100
  labels     correct iff a normalised TriviaQA alias occurs in the normalised
             answer (the standard containment rule for generative QA);
             GUARDIAN's ROUGE-L >= 0.3 rule is unsuitable here because a
             correct one-word entity inside a sentence scores below 0.3.
             Answers are saved, so any other rule can be applied offline.
  batching   batched generation with left padding; the hidden-state pass is a
             separate unbatched-equivalent forward with right padding, read
             at each row's own last real token.
Label convention as GUARDIAN: 1 = hallucinated (incorrect), 0 = correct.
"""
import json
import os
import re
import string
import time
from pathlib import Path

import numpy as np
import torch

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
MODEL_REVISION = "63a8b081895390a26e140280378bc85ec8bce07a"
SMOKE = False  # smoke test (96 questions) passed 2026-10-12
N_SHARDS, SHARD_SIZE = (1, 96) if SMOKE else (11, 700)
N_TOTAL = N_SHARDS * SHARD_SIZE
SEED = 20261013
N_LAYERS = 32
MAX_NEW_TOKENS = 32
GEN_BATCH = 32
HS_BATCH = 16
PROMPT_TEMPLATE = ("<s>[INST] Answer the following question concisely and factually."
                   "\n\nQuestion: {question} [/INST]")
OUT = Path("/kaggle/working")
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# ── labels ─────────────────────────────────────────────────────────────────
def _variants(s):
    """TriviaQA-style normalisation (lower, articles out, whitespace collapsed),
    in two forms: punctuation deleted, and punctuation replaced by a space.
    TriviaQA's stored aliases use the second ("merrill s marauders"), model
    answers usually need the first, so matching is done across both."""
    s = s.lower()
    p = set(string.punctuation)
    out = set()
    for v in ("".join(ch for ch in s if ch not in p),
              "".join(" " if ch in p else ch for ch in s)):
        v = re.sub(r"\b(a|an|the)\b", " ", v)
        out.add(" ".join(v.split()))
    return out


def normalize(s):
    return sorted(_variants(s))[0]


def is_correct(answer, aliases):
    """Whole-token containment of any alias variant in any answer variant."""
    A = [f" {v} " for v in _variants(answer)]
    L = set().union(*[_variants(a) for a in aliases if a]) if aliases else set()
    return any(f" {l} " in a for l in L if l for a in A)


# ── data ───────────────────────────────────────────────────────────────────
def load_questions():
    """TOP-UP RUN. The first run drew from rc.nocontext validation, which repeats
    questions; offline deduplication left 14 independent samples. This run
    supplies 11 more from the TRAIN split, deduplicated here by question id and
    by normalised text, and excluding every question that appears anywhere in
    validation, so no question can be shared with the first 14 samples."""
    from datasets import load_dataset
    val = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="validation")
    val_ids = set(val["question_id"])
    val_txt = {" ".join(q.lower().split()) for q in val["question"]}
    tr = load_dataset("mandarjoshi/trivia_qa", "rc.nocontext", split="train")
    rng = np.random.default_rng(SEED)
    seen_id, seen_txt, rows = set(), set(), []
    for i in rng.permutation(len(tr)):
        r = tr[int(i)]
        t = " ".join(r["question"].lower().split())
        if r["question_id"] in val_ids or t in val_txt or r["question_id"] in seen_id or t in seen_txt:
            continue
        seen_id.add(r["question_id"]); seen_txt.add(t)
        al = set(r["answer"]["normalized_aliases"]) | {r["answer"]["normalized_value"]}
        rows.append({"qid": r["question_id"], "question": r["question"],
                     "aliases": sorted(a for a in al if a)})
        if len(rows) == N_TOTAL:
            break
    assert len(rows) == N_TOTAL, f"only {len(rows)} unique questions found"
    log(f"loaded {len(rows)} unique train questions disjoint from validation (seed {SEED})")
    return rows


# ── model ──────────────────────────────────────────────────────────────────
def load_model():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=MODEL_REVISION, torch_dtype=torch.float16,
        device_map="auto", low_cpu_mem_usage=True)
    model.eval()
    for d in range(torch.cuda.device_count()):
        log(f"GPU{d} {torch.cuda.get_device_name(d)} "
            f"{torch.cuda.memory_allocated(d)/1e9:.1f} GB allocated")
    return tok, model


def register_hooks(model, store):
    def make(i):
        def hook(module, inp, out):
            store[i] = out[0] if isinstance(out, tuple) else out
        return hook
    for i, layer in enumerate(model.model.layers):
        layer.register_forward_hook(make(i))


@torch.no_grad()
def generate(tok, model, questions):
    tok.padding_side = "left"
    prompts = [PROMPT_TEMPLATE.format(question=q) for q in questions]
    enc = tok(prompts, return_tensors="pt", padding=True).to(model.device)
    out = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=tok.pad_token_id)
    seqs, answers = [], []
    plen = enc["input_ids"].shape[1]
    for r in range(len(prompts)):
        prompt_ids = enc["input_ids"][r][enc["attention_mask"][r].bool()].tolist()
        gen = out[r, plen:].tolist()
        # cut at the first EOS (also the pad token here) -- that strips the
        # trailing EOS exactly as GUARDIAN's single-sample loop did
        if tok.eos_token_id in gen:
            gen = gen[:gen.index(tok.eos_token_id)]
        seqs.append(prompt_ids + gen)
        answers.append(tok.decode(gen, skip_special_tokens=True).strip())
    return seqs, answers


@torch.no_grad()
def hidden_states(tok, model, seqs, store):
    """Last-real-token state at every decoder layer, right-padded batch."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(seqs), L), dtype=torch.long)
    for r, s in enumerate(seqs):
        ids[r, :len(s)] = torch.tensor(s); att[r, :len(s)] = 1
    ids, att = ids.to(model.device), att.to(model.device)
    model(input_ids=ids, attention_mask=att)
    last = torch.tensor([len(s) - 1 for s in seqs])
    out = np.empty((len(seqs), N_LAYERS, 4096), dtype=np.float16)
    for l in range(N_LAYERS):
        h = store[l]
        out[:, l, :] = h[torch.arange(len(seqs), device=h.device),
                         last.to(h.device)].float().cpu().numpy().astype(np.float16)
    return out


def main():
    t0 = time.time()
    rows = load_questions()
    tok, model = load_model()
    store = {}
    register_hooks(model, store)
    meta = {"model": MODEL_ID, "revision": MODEL_REVISION, "dataset": "mandarjoshi/trivia_qa rc.nocontext train, unique, disjoint from validation",
            "seed": SEED, "n_shards": N_SHARDS, "shard_size": SHARD_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS, "label_rule": "alias containment; 1 = incorrect",
            "prompt_template": PROMPT_TEMPLATE, "shards": []}
    for sh in range(N_SHARDS):
        path = OUT / f"shard_{sh:02d}.npz"
        if path.exists():
            log(f"shard {sh} exists, skipping"); continue
        part = rows[sh * SHARD_SIZE:(sh + 1) * SHARD_SIZE]
        H = np.empty((len(part), N_LAYERS, 4096), dtype=np.float16)
        labels, answers = np.empty(len(part), dtype=np.int64), []
        for b in range(0, len(part), GEN_BATCH):
            chunk = part[b:b + GEN_BATCH]
            seqs, ans = generate(tok, model, [c["question"] for c in chunk])
            for k in range(0, len(seqs), HS_BATCH):
                H[b + k:b + k + HS_BATCH] = hidden_states(tok, model, seqs[k:k + HS_BATCH], store)
            for j, (c, a) in enumerate(zip(chunk, ans)):
                labels[b + j] = 0 if is_correct(a, c["aliases"]) else 1
            answers += ans
        np.savez_compressed(path, hidden_states=H, labels=labels)
        with open(OUT / f"shard_{sh:02d}_answers.json", "w") as f:
            json.dump([{"qid": c["qid"], "question": c["question"], "answer": a,
                        "aliases": c["aliases"], "label": int(l)}
                       for c, a, l in zip(part, answers, labels)], f)
        meta["shards"].append({"shard": sh, "n": len(part),
                               "hallucination_rate": float(labels.mean())})
        log(f"shard {sh+1}/{N_SHARDS} done, hall_rate={labels.mean():.3f}, "
            f"elapsed {(time.time()-t0)/60:.1f} min")
        with open(OUT / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    meta["total_minutes"] = (time.time() - t0) / 60
    with open(OUT / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log("done")


if __name__ == "__main__":
    main()
