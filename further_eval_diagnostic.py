"""
Cloze diagnostic, corrected.

Fixes vs the previous version:
  1. No trailing space on prefixes. SentencePiece merges a trailing space into
     the next word, so tokenizing prefix-alone vs prefix+completion disagreed
     and produced empty slices -> nan. Completions now carry their own leading
     space, and the prefix tokenization is verified to be an actual prefix of
     the joint tokenization.
  2. Reports SUM of logprobs as the primary score, not the length-normalised
     mean. Mean dilutes the one discriminating token among predictable ones
     (e.g. the year token after a month name), which washed out probe 2.
  3. Candidates kept as close to single-token-different as possible so the
     score reflects the fact, not incidental phrasing.
  4. Reports the MARGIN between the top two candidates. A "correct" ranking
     won by 0.008 is a coin flip, not knowledge -- the margin is the number
     to read, not the YES/no.
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

MODELS = {
    "BASE": "google/gemma-3-1b-pt",
    "v5":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v5",
    "v6":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v6",
    "v7":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v7",
    "v8":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v8",
    "v9":   "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v9",
    "sft_v1": "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_sft/models/gemma_3_1B_ncert_sft_v1",
}

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Prefixes must NOT end with a space. Completions must START with a space.
# First candidate is always the TRUE one.
PROBES = [
    (
        "In the listening-exercise letter, the last thing Jaya had sent Arti before this letter was a",
        [" birthday card", " postcard", " letter", " photograph"],
    ),
    (
        "Jaya's letter to Arti is dated 25 January 2006. She says the birthday card she sent Arti went out in the month of",
        [" September", " November", " January", " March"],
    ),
    (
        "In his letter to God, Lencho asked for exactly",
        [" one hundred pesos", " seventy pesos", " one thousand pesos", " ten pesos"],
    ),
    (
        "In 'Madam Rides the Bus', Valli tells the elderly man that she has paid her",
        [" thirty paise", " sixty paise", " fifteen paise", " twenty paise"],
    ),
]


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
    model.eval()
    return tok, model


def score(tok, model, prefix, completion):
    """Returns (sum_logprob, mean_logprob, n_tokens) for `completion` given `prefix`."""
    prefix_ids = tok(prefix, return_tensors="pt").input_ids
    full_ids = tok(prefix + completion, return_tensors="pt").input_ids

    n_prefix = prefix_ids.shape[-1]
    # Verify prefix tokenization really is a prefix of the joint one; if the
    # tokenizer merged across the boundary, walk back to the true common length
    # instead of silently scoring the wrong span.
    common = 0
    for a, b in zip(prefix_ids[0].tolist(), full_ids[0].tolist()):
        if a != b:
            break
        common += 1
    if common != n_prefix:
        n_prefix = common

    n_completion = full_ids.shape[-1] - n_prefix
    if n_completion <= 0:
        raise ValueError(f"No completion tokens for {completion!r} -- check spacing.")

    full_ids = full_ids.to(DEVICE)
    with torch.inference_mode():
        logits = model(full_ids).logits

    logprobs = F.log_softmax(logits[0, :-1].float(), dim=-1)
    targets = full_ids[0, 1:]
    comp = logprobs[n_prefix - 1:].gather(1, targets[n_prefix - 1:].unsqueeze(1)).squeeze(1)
    total = comp.sum().item()
    return total, total / len(comp), len(comp)


def main():
    results = [dict() for _ in PROBES]
    for label, path in MODELS.items():
        print(f"Loading {label} ...")
        tok, model = load(path)
        for i, (prefix, candidates) in enumerate(PROBES):
            results[i][label] = {c: score(tok, model, prefix, c) for c in candidates}
        del model
        torch.cuda.empty_cache()

    for i, (prefix, candidates) in enumerate(PROBES):
        truth = candidates[0]
        print(f"\n{'=' * 80}")
        print(f"PROBE {i + 1}: ...{prefix[-72:]}")
        print(f"TRUTH:{truth}")
        print("=" * 80)
        print(f"{'candidate':<24}" + "".join(f"{l:>18}" for l in MODELS))
        for c in candidates:
            mark = "*" if c == truth else " "
            row = f"{mark}{c.strip():<23}"
            for l in MODELS:
                row += f"{results[i][l][c][0]:>18.3f}"
            print(row)

        rank_row = f"{'-> truth ranked #1?':<24}"
        margin_row = f"{'-> margin over 2nd':<24}"
        for l in MODELS:
            ranked = sorted(results[i][l].items(), key=lambda kv: -kv[1][0])
            best, second = ranked[0], ranked[1]
            rank_row += f"{('YES' if best[0] == truth else 'no'):>18}"
            # margin is positive when truth leads, negative when it trails
            truth_score = results[i][l][truth][0]
            rival = second[1][0] if best[0] == truth else best[1][0]
            margin_row += f"{truth_score - rival:>18.3f}"
        print(rank_row)
        print(margin_row)

    print("\nRead the MARGIN row, not the YES/no. A margin under ~0.5 is noise.")


if __name__ == "__main__":
    main()