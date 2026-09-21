"""
Score all checkpoints against eval_probes.jsonl.

Reports, per checkpoint:
  - overall accuracy (% of probes where the truth ranks #1)
  - per-category accuracy (numeric / relational / name / vocabulary)
  - mean margin (truth score minus best rival) -- the real confidence signal
  - BOTH sum-logprob and mean-logprob rankings

Why both: sum-logprob is biased against longer completions (more negative terms
accumulate), mean-logprob dilutes the one discriminating token among predictable
ones. Where the two disagree, the probe's candidate lengths are uneven and the
result should not be trusted. The script prints that disagreement count.
"""
import json, argparse, os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

MODELS = {
    "BASE": "google/gemma-3-1b-pt",
    "v5":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v5",
    "v6":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v6",
    "v7":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v7",
    "v8":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v8",
    "v9":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v9",
    "sft_v1": "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_sft/models/gemma_3_1B_ncert_sft_v1",
}

# MODELS = {
#     "BASE": "Qwen/Qwen3-1.7B-Base",
#     "v1":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/qwen_3_1point7B_base_ncert_cpt_v1",
#     "v2":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/qwen_3_1point7B_base_ncert_cpt_v2",
#     "v3":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/qwen_3_1point7B_base_ncert_cpt_v3",
# }


def load_probes(path):
    probes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                probes.append(json.loads(line))
    # fail loudly on the formatting bugs that silently broke earlier probe sets
    for p in probes:
        assert not p["prefix"].endswith(" "), f"{p['probe_id']}: prefix ends with a space"
        for c in [p["truth"]] + p["distractors"]:
            assert c.startswith(" "), f"{p['probe_id']}: candidate {c!r} lacks a leading space"
    return probes


def score(tok, model, prefix, completion):
    """Returns (sum_logprob, mean_logprob) for `completion` given `prefix`."""
    prefix_ids = tok(prefix, return_tensors="pt").input_ids
    full_ids = tok(prefix + completion, return_tensors="pt").input_ids

    # Guard against SentencePiece merging across the prefix/completion boundary:
    # walk forward to the real common prefix length rather than trusting len(prefix_ids).
    common = 0
    for a, b in zip(prefix_ids[0].tolist(), full_ids[0].tolist()):
        if a != b:
            break
        common += 1
    n_prefix = common
    if full_ids.shape[-1] - n_prefix <= 0:
        raise ValueError(f"No completion tokens for {completion!r}")

    full_ids = full_ids.to(DEVICE)
    with torch.inference_mode():
        logits = model(full_ids).logits
    logprobs = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = full_ids[0, 1:]
    comp = logprobs[n_prefix - 1:].gather(1, targets[n_prefix - 1:].unsqueeze(1)).squeeze(1)
    total = comp.sum().item()
    return total, total / len(comp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", default="eval_probes.jsonl")
    args = ap.parse_args()

    probes = load_probes(args.probes)
    print(f"Loaded {len(probes)} probes from {args.probes}\n")

    summary = {}
    for label, path in MODELS.items():
        if not (path.startswith("google/") or path.startswith("Qwen/") or os.path.isdir(path)):
            print(f"  skipping {label}: {path} not found")
            continue
        print(f"Scoring {label} ...")
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
        model.eval()

        correct_sum = correct_mean = 0
        disagree = 0
        margins = []
        by_cat = defaultdict(lambda: [0, 0])  # [correct, total]
        wrong = []

        for p in probes:
            cands = [p["truth"]] + p["distractors"]
            s = {c: score(tok, model, p["prefix"], c) for c in cands}

            best_sum = max(s, key=lambda c: s[c][0])
            best_mean = max(s, key=lambda c: s[c][1])
            if best_sum != best_mean:
                disagree += 1

            hit = best_sum == p["truth"]
            correct_sum += hit
            correct_mean += best_mean == p["truth"]

            rival = max((c for c in cands if c != p["truth"]), key=lambda c: s[c][0])
            margins.append(s[p["truth"]][0] - s[rival][0])

            by_cat[p["category"]][1] += 1
            by_cat[p["category"]][0] += hit
            if not hit:
                wrong.append((p["probe_id"], p["category"], p["truth"].strip(), best_sum.strip()))

        n = len(probes)
        summary[label] = {
            "acc": 100 * correct_sum / n,
            "acc_mean": 100 * correct_mean / n,
            "margin": sum(margins) / n,
            "by_cat": {k: 100 * v[0] / v[1] for k, v in by_cat.items()},
            "disagree": disagree,
            "wrong": wrong,
        }
        del model
        torch.cuda.empty_cache()

    cats = ["numeric", "relational", "name", "vocabulary"]
    print(f"\n{'=' * 84}\nRESULTS  (random-guess baseline = 25%)\n{'=' * 84}")
    print(f"{'metric':<22}" + "".join(f"{l:>12}" for l in summary))
    print(f"{'accuracy (sum)':<22}" + "".join(f"{summary[l]['acc']:>11.1f}%" for l in summary))
    print(f"{'accuracy (mean)':<22}" + "".join(f"{summary[l]['acc_mean']:>11.1f}%" for l in summary))
    print(f"{'mean margin':<22}" + "".join(f"{summary[l]['margin']:>12.3f}" for l in summary))
    print(f"{'sum/mean disagree':<22}" + "".join(f"{summary[l]['disagree']:>12}" for l in summary))
    print("-" * 84)
    for c in cats:
        print(f"{'  ' + c:<22}" + "".join(f"{summary[l]['by_cat'].get(c, 0):>11.1f}%" for l in summary))

    last = list(summary)[-1]
    print(f"\nProbes {last} got wrong:")
    for pid, cat, truth, picked in summary[last]["wrong"]:
        print(f"  {pid} [{cat}]  truth={truth!r}  picked={picked!r}")

    print("\nWith 50 probes, a gap under ~10 points between checkpoints is within noise.")
    print("Read per-category accuracy: a numeric score near 25% is a capacity finding, not a data one.")


if __name__ == "__main__":
    main()