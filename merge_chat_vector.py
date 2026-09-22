"""
Chat-vector merge: give the CPT model back the instruct model's behaviour, with no training.

    merged = cpt_v8 + alpha * (gemma-3-1b-it - gemma-3-1b-pt)

(gemma-3-1b-it - gemma-3-1b-pt) is everything Google's post-training added on top of the
base model: chat format, instruction following, markdown, multi-turn, safety alignment.
Our CPT model started from the same base, so the two weight changes can be added. This is
the method from "Chat Vector" (Huang et al., ACL 2024), which did CPT on a new language
and then added this vector to get a chat model without any SFT.

It is an experiment, not a guaranteed win: the two changes can interfere. Several alphas
are saved so the tradeoff can be measured, not guessed.

Usage
  python merge_chat_vector.py                       # alphas 0.5, 0.75, 1.0
  python merge_chat_vector.py --alpha 0.75          # one alpha
  python merge_chat_vector.py --cpt <other ckpt>    # e.g. a lighter CPT run if v8 clashes
"""
import argparse, os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

ROOT = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training"
PT = "google/gemma-3-1b-pt"
IT = "google/gemma-3-1b-it"
CPT = f"{ROOT}/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v8"
OUT_ROOT = f"{ROOT}/ncert_model_merge/models"


def load_sd(path):
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    ap.add_argument("--cpt", default=CPT)
    a = ap.parse_args()

    print("loading base, instruct and CPT weights (fp32)...")
    pt_sd, it_sd, cpt_sd = load_sd(PT), load_sd(IT), load_sd(a.cpt)

    # all three must be the same architecture with identical parameter names and shapes
    assert pt_sd.keys() == it_sd.keys() == cpt_sd.keys(), "parameter names differ - not the same architecture"
    for k in pt_sd:
        assert pt_sd[k].shape == it_sd[k].shape == cpt_sd[k].shape, f"shape mismatch at {k}"

    # the chat vector, computed once
    chat_vec = {k: it_sd[k] - pt_sd[k] for k in pt_sd if pt_sd[k].is_floating_point()}
    del pt_sd, it_sd

    # how large the two changes are, relative to each other - useful when reading results
    cpt_base = load_sd(PT)
    cpt_norm = sum(((cpt_sd[k] - cpt_base[k]).float() ** 2).sum() for k in chat_vec) ** 0.5
    chat_norm = sum((chat_vec[k].float() ** 2).sum() for k in chat_vec) ** 0.5
    del cpt_base
    print(f"size of CPT change  |cpt - pt| = {cpt_norm:.2f}")
    print(f"size of chat vector |it - pt|  = {chat_norm:.2f}")

    tok = AutoTokenizer.from_pretrained(IT)          # official Gemma chat template
    gen_cfg = GenerationConfig.from_pretrained(IT)   # stops on <end_of_turn> as well as <eos>

    for alpha in a.alpha:
        merged = {k: (cpt_sd[k] + alpha * chat_vec[k]) if k in chat_vec else cpt_sd[k] for k in cpt_sd}
        model = AutoModelForCausalLM.from_pretrained(a.cpt, dtype=torch.float32)
        missing, unexpected = model.load_state_dict(merged, strict=False)
        assert not unexpected, f"unexpected keys: {unexpected[:5]}"
        model = model.to(torch.bfloat16)
        model.generation_config = gen_cfg

        out = os.path.join(OUT_ROOT, f"gemma_3_1B_ncert_cv8_chatvec_a{alpha}")
        os.makedirs(out, exist_ok=True)
        model.save_pretrained(out)
        tok.save_pretrained(out)
        print(f"alpha={alpha} -> {out}")
        del model, merged


if __name__ == "__main__":
    main()