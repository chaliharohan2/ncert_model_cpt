from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_NAME = "google/gemma-3-1b-pt"
CPT_PATH = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v2"
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path).to(DEVICE)
    model.eval()
    return tok, model


def generate(tok, model, prompt, max_new_tokens=60):
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    input_len = inputs["input_ids"].shape[-1]
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return tok.decode(out[0][input_len:], skip_special_tokens=True)


# Each probe tests a different failure mode, not just "does it answer correctly":
# - completion: did training move the weights toward this text at all (mechanics check)
# - few_shot_qa_famous: can it answer via general web knowledge even with zero CPT
#   (this story is public-domain-ish and widely anthologized, so expect the BASE
#   model to do okay here too — a good result on this alone proves little)
# - few_shot_qa_curriculum_specific: a detail that only exists in this NCERT chapter's
#   embedded listening-exercise text, unlikely to be duplicated anywhere else on the
#   web — the real test of whether CPT injected something new
# - paraphrase_robustness_*: the same fact asked two different ways — if the CPT model
#   nails the original phrasing but fails the reworded one, that's evidence you need
#   more paraphrase diversity in the data, not more epochs of the same phrasings
PROBES = {
    "completion": (
        "The house — the only one in the entire valley — sat\n"
        "on the crest of a low hill. From this height one"
    ),
    "few_shot_qa_famous": (
        "Q: What is the capital of France?\nA: Paris.\n\n"
        "Q: How much money did Lencho ask God for in his letter?\nA:"
    ),
    "few_shot_qa_curriculum_specific": (
        "Q: What is the capital of France?\nA: Paris.\n\n"
        "Q: In the listening exercise letter in this chapter, how long had it been "
        "since Jaya last wrote to Arti, and what had she sent at that time?\nA:"
    ),
    "paraphrase_robustness_original": (
        "Q: What is the capital of France?\nA: Paris.\n\n"
        "Q: How much money did Lencho ask God for?\nA:"
    ),
    "paraphrase_robustness_reworded": (
        "Q: What is the capital of France?\nA: Paris.\n\n"
        "Q: What amount did Lencho request in his letter to God?\nA:"
    ),
}


def run_all(label, tok, model):
    print(f"\n=== {label} ===")
    for name, prompt in PROBES.items():
        print(f"\n--- {name} ---\n{generate(tok, model, prompt)}")


if __name__ == "__main__":
    tok_base, model_base = load(MODEL_NAME)
    run_all("BASE MODEL (no CPT)", tok_base, model_base)
    del model_base
    torch.cuda.empty_cache()

    tok_cpt, model_cpt = load(CPT_PATH)
    run_all("CPT MODEL", tok_cpt, model_cpt)