"""
Stage B SFT on top of the CPT v8 checkpoint.

Fixes vs the previous draft:
  1. Loads the CPT v8 checkpoint, not google/gemma-3-1b-pt. SFT on the raw base
     model would discard every CPT run.
  2. Writes to a NEW directory. The old script saved into the v8 directory and
     would have overwritten the best CPT checkpoint.
  3. Supplies a chat template with {% generation %} markers. assistant_only_loss
     only works if the template marks which tokens are the assistant's; Gemma's
     stock template does not, so the loss mask would be empty.
  4. Puts <end_of_turn> INSIDE the generation block, so the model is trained to
     emit it. That is what teaches it to stop instead of inventing follow-up turns.
  5. Excludes needs_human_review records (unreviewed authored exercise answers)
     for this first run.
  6. Saves every epoch, so each checkpoint can be cloze-scored for forgetting.
"""
import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

ROOT = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training"
CPT_CHECKPOINT = f"{ROOT}/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v8"   
OUTPUT_DIR     = f"{ROOT}/ncert_model_sft/models/gemma_3_1B_ncert_sft_v1"      
DATASET_PATH   = f"{ROOT}/dataset/sft_set.jsonl"
MAX_LENGTH = 2048  # longest assistant turn is ~230 words; 8000 was far larger than needed

# Gemma 3 turn format, plus generation markers around the assistant turn.
# <end_of_turn> sits inside {% generation %} so it receives loss and the model
# learns to stop. The trailing newline sits outside and is not trained.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] | trim }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<start_of_turn>model\n"
    "{% generation %}{{ message['content'] | trim }}<end_of_turn>{% endgeneration %}"
    "{{ '\\n' }}"  # newline emitted as output: jinja's trim_blocks deletes a literal one right after a block tag
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def check_mask(tokenizer, messages):
    """Print exactly which tokens will carry loss. Run this before trusting training."""
    enc = tokenizer.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_assistant_tokens_mask=True
    )
    ids, mask = enc["input_ids"], enc["assistant_masks"]
    trained = tokenizer.decode([t for t, m in zip(ids, mask) if m])
    n_bos = ids.count(tokenizer.bos_token_id)
    print("---- mask check ----")
    print("full sequence :", repr(tokenizer.decode(ids)))
    print("trained tokens:", repr(trained))
    print("BOS count     :", n_bos)
    assert sum(mask) > 0, "assistant mask is empty - template generation markers not working"
    assert trained.rstrip().endswith("<end_of_turn>"), "<end_of_turn> is not being trained"
    assert n_bos == 1, f"expected exactly 1 BOS, found {n_bos}"
    print("mask check passed\n")


def main():
    assert os.path.abspath(OUTPUT_DIR) != os.path.abspath(CPT_CHECKPOINT), \
        "OUTPUT_DIR must differ from CPT_CHECKPOINT or v8 gets overwritten"

    tokenizer = AutoTokenizer.from_pretrained(CPT_CHECKPOINT)
    tokenizer.chat_template = CHAT_TEMPLATE
    for tok in ("<start_of_turn>", "<end_of_turn>"):
        assert tokenizer.convert_tokens_to_ids(tok) != tokenizer.unk_token_id, f"{tok} missing from vocab"

    # fp32 weights; bf16=True below gives autocast with fp32 master weights
    model = AutoModelForCausalLM.from_pretrained(CPT_CHECKPOINT)

    # stop generation at <end_of_turn> as well as <eos>, saved with the model
    eot = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    model.generation_config.eos_token_id = [tokenizer.eos_token_id, eot]

    ds = load_dataset("json", data_files=DATASET_PATH, split="train")
    n_all = len(ds)
    # ds = ds.filter(lambda r: not r["needs_human_review"])
    # print(f"kept {len(ds)} / {n_all} records (dropped unreviewed exercise answers)")
    ds = ds.select_columns(["messages"])

    check_mask(tokenizer, ds[0]["messages"])
    
    args = SFTConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=1e-5,            # ~5x below the CPT optimum, to protect CPT knowledge
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch 16; batch 2 was noisy in CPT
        lr_scheduler_type="cosine",
        warmup_steps=0.03,             
        logging_steps=10,
        save_strategy="epoch",         # checkpoint per epoch -> cloze-score each for forgetting
        save_total_limit=3,
        assistant_only_loss=True,
        bf16=True,
        max_length=MAX_LENGTH,
        seed=64,
        report_to="none",
    )

    trainer = SFTTrainer(model=model, train_dataset=ds, args=args, processing_class=tokenizer)
    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)  # persists the chat template for inference


if __name__ == "__main__":
    main()