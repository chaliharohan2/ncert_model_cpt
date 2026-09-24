"""
Step 2: one training run that mixes all three streams, starting from an instruct model.

The previous path was CPT then SFT, in sequence, from the base model. It put the
knowledge in (cloze 44% -> 84%) but the finished model can do nothing except answer
questions about this one textbook: no instruction following, no conversation, no
refusals. The chat-vector merge was meant to hand those back and failed decisively
(|cpt-pt| = 41.5 against |it-pt| = 6749.7; pt and it are not linearly connected, so
no alpha and no merge method will bridge them).

So this script does the two jobs at once, from gemma-3-1b-it, with replay holding
the instruct behaviour in place:

  stream        loss on            purpose
  ------------  -----------------  --------------------------------------------
  cpt_pool      every token        puts the book knowledge into the weights
  sft_set_v2    assistant turns    formats, multi-turn, "the book doesn't say"
  replay_set    assistant turns    keeps the instruct behaviour you started with

All three are shuffled into one dataset, so every batch contains some of each. A
batch made only of book text is a batch that pushes the instruct behaviour out.

How hard to train. The CPT sweeps found the knowledge landing best at a training
intensity (LR x epochs) of about 7.5e-4 - v8 was 5e-5 at that intensity. That
number is the one to keep in mind here, and it creates a real tension: a short,
gentle run (say 1.5e-5 x 3 epochs = 4.5e-5) is roughly a seventeenth of the
exposure that got cloze to 84%, and the book will not land. A run at the full
7.5e-4 is the one most likely to wear away the instruct behaviour. Replay is what
lets you push harder than you otherwise could, but nobody can tell you in advance
where this model's balance point is. So the defaults are one long run with a
checkpoint every epoch:

    --lr 2e-5 --epochs 10     (intensity 2e-4 by the end, ~a quarter of CPT's)

and after EACH epoch you run the cloze scorer (knowledge - compare against v8's
84% and against gemma-3-1b-it's own cloze score, which is the real starting
point here, not the base model's 44%) and eval_retention.py (what you kept). You
want the last epoch before retention drops. If cloze is still climbing when
retention starts to fall, raise --cpt-repeat rather than the LR: it adds book
exposure without adding pressure on everything else. If cloze plateaus well below
v8 with retention intact, the next run is --lr 5e-5.

Usage
  # full fine-tune, the default path
  python train_mixed.py                       # 2e-5, 10 epochs, save every epoch

  # LoRA instead - much cheaper, forgets less, learns less, and wants a higher LR.
  # This is also how you test the per-subject-adapter idea: one adapter per subject
  # over one base model.
  python train_mixed.py --lora --lr 2e-4 --epochs 10

  # sanity pass: build the data, print the loss masks, train on 50 steps, stop
  python train_mixed.py --dry-run
"""
import argparse
import json
import os
import random

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

ROOT = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training"
DEFAULTS = dict(
    model=f"google/gemma-3-1b-it",
    cpt=f"{ROOT}/dataset/cpt_pool.jsonl",
    sft=f"{ROOT}/dataset/sft_set_v2_fixed.jsonl",
    replay=f"{ROOT}/dataset/replay_set.jsonl",
    out=f"{ROOT}/ncert_model_mixed/models/gemma_3_1B_ncert_mixed_v1",
)

# The same template as train_sft.py. {% generation %} is what makes
# return_assistant_tokens_mask work; Gemma's stock template has no such marker, so
# without this the assistant mask comes back empty and nothing is trained.
# The trailing newline is emitted as output because Jinja's trim_blocks deletes a
# literal newline written straight after a block tag - which is invisible in
# single-turn data and wrong in every multi-turn conversation.
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] | trim }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<start_of_turn>model\n"
    "{% generation %}{{ message['content'] | trim }}<end_of_turn>{% endgeneration %}"
    "{{ '\\n' }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class MixedDataset(TorchDataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_cpt_examples(tok, path, block, text_key=None):
    """Raw book text -> fixed-length windows, loss on every token.

    Each document gets its own BOS and is windowed separately rather than packed
    across documents, so no window straddles two chapters. add_special_tokens=False
    because the BOS is added explicitly - Gemma's tokenizer would otherwise add a
    second one and the model would see a BOS it never sees at inference.
    """
    rows = load_jsonl(path)
    if text_key is None:
        for k in ("text", "raw_text", "content", "chunk"):
            if k in rows[0]:
                text_key = k
                break
        else:
            raise SystemExit(f"no text field in {path}; columns are {list(rows[0])}. "
                             f"Pass --cpt-text-key.")
    bos = tok.bos_token_id
    out, n_tok = [], 0
    for r in rows:
        ids = [bos] + tok(r[text_key], add_special_tokens=False)["input_ids"]
        n_tok += len(ids)
        windows = [ids[s:s + block] for s in range(0, len(ids), block)]
        # A short trailing stub is merged back into the window before it rather than
        # dropped. Dropping would quietly leave the last sentences of a chunk out of
        # training, and coverage is the whole point of this stream.
        if len(windows) > 1 and len(windows[-1]) < 32:
            windows[-2] = windows[-2] + windows.pop()
        for w in windows:
            out.append({"input_ids": w, "labels": list(w), "stream": "cpt"})
    print(f"  {path}: {len(rows)} documents, {n_tok:,} tokens -> {len(out)} windows "
          f"(field {text_key!r}); every token is covered exactly once")
    return out


def build_chat_examples(tok, path, stream, max_length):
    """Chat records -> templated ids with loss on the assistant turns only."""
    rows = load_jsonl(path)
    out, dropped = [], 0
    for r in rows:
        enc = tok.apply_chat_template(r["messages"], tokenize=True, return_dict=True,
                                      return_assistant_tokens_mask=True)
        ids, mask = enc["input_ids"], enc["assistant_masks"]
        if len(ids) > max_length:
            dropped += 1
            continue
        if sum(mask) == 0:
            raise SystemExit(f"{r.get('id')}: empty assistant mask - the chat template "
                             f"lost its {{% generation %}} markers")
        out.append({"input_ids": ids,
                    "labels": [t if m else -100 for t, m in zip(ids, mask)],
                    "stream": stream,
                    "tag": r.get("tag") or r.get("type")})
    if dropped:
        print(f"  {path}: dropped {dropped} records longer than {max_length} tokens")
    return out


class Collator:
    """Pads right, pads labels with -100.

    Labels are NOT shifted here. The model shifts internally, so labels line up with
    input_ids position for position - the same convention DataCollatorForLanguageModeling
    uses.
    """

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, labels, attn = [], [], []
        for b in batch:
            k = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad_id] * k)
            labels.append(b["labels"] + [-100] * k)
            attn.append([1] * len(b["input_ids"]) + [0] * k)
        return {"input_ids": torch.tensor(ids),
                "labels": torch.tensor(labels),
                "attention_mask": torch.tensor(attn)}


def show(tok, ex, title):
    ids, labels = ex["input_ids"], ex["labels"]
    trained = tok.decode([t for t, l in zip(ids, labels) if l != -100])
    print(f"--- {title} ({len(ids)} tokens, {sum(1 for l in labels if l != -100)} carry loss)")
    print("    full   :", repr(tok.decode(ids)[:260]))
    print("    trained:", repr(trained[:260]))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULTS["model"])
    ap.add_argument("--cpt", default=DEFAULTS["cpt"])
    ap.add_argument("--sft", default=DEFAULTS["sft"])
    ap.add_argument("--replay", default=DEFAULTS["replay"])
    ap.add_argument("--out", default=DEFAULTS["out"])
    ap.add_argument("--cpt-text-key", default=None)

    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=10)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=8)   # effective batch 16
    ap.add_argument("--block", type=int, default=1024, help="CPT window length")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=64)

    ap.add_argument("--replay-frac", type=float, default=0.25,
                    help="replay as a share of all examples. Below ~0.15 the instruct "
                         "behaviour starts going; above ~0.35 the book knowledge lands "
                         "more slowly.")
    ap.add_argument("--cpt-repeat", type=float, default=1.0,
                    help="how many times the raw-text windows are repeated per epoch. "
                         "Raise it if the book knowledge is not landing.")

    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)

    ap.add_argument("--dry-run", action="store_true",
                    help="build the data, print the masks, run 50 steps, stop")
    a = ap.parse_args()

    assert os.path.abspath(a.out) != os.path.abspath(a.model), \
        "--out must differ from --model, or the starting model is overwritten"
    for p in (a.cpt, a.sft, a.replay):
        if not os.path.exists(p):
            raise SystemExit(f"missing input: {p}")

    random.seed(a.seed)
    torch.manual_seed(a.seed)

    tok = AutoTokenizer.from_pretrained(a.model)

    # The starting model has its own template. Ours differs only by the generation
    # markers, so the rendered text must be byte-identical - otherwise training
    # teaches a format the model was never instruct-tuned on.
    probe = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello."},
             {"role": "user", "content": "And again?"}]
    if tok.chat_template:
        theirs = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True)
        ours = tok.apply_chat_template(probe, tokenize=False, add_generation_prompt=True,
                                       chat_template=CHAT_TEMPLATE)
        if theirs != ours:
            print("!! the training template does not render the same text as the model's own:")
            print("   model's:", repr(theirs))
            print("   ours   :", repr(ours))
            raise SystemExit("fix the template before training")
        print("template check: renders identically to the model's own template")
    tok.chat_template = CHAT_TEMPLATE
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print("\nbuilding data ...")
    cpt = build_cpt_examples(tok, a.cpt, a.block, a.cpt_text_key)
    if a.cpt_repeat != 1.0:
        k = int(len(cpt) * a.cpt_repeat)
        cpt = (cpt * (int(a.cpt_repeat) + 1))[:k]
    sft = build_chat_examples(tok, a.sft, "sft", a.max_length)
    replay_all = build_chat_examples(tok, a.replay, "replay", a.max_length)

    # trim replay to the requested share of the final mix
    base = len(cpt) + len(sft)
    want = int(base * a.replay_frac / max(1e-9, 1 - a.replay_frac))
    random.shuffle(replay_all)
    replay = replay_all[:want]
    if len(replay) < want:
        print(f"note: replay set has {len(replay_all)} usable records but {want} were "
              f"wanted for --replay-frac {a.replay_frac}. Actual share will be "
              f"{len(replay) / (base + len(replay)):.1%}. Generate more with build_replay_set.py.")

    examples = cpt + sft + replay
    random.shuffle(examples)

    tokens = {}
    for e in examples:
        s = e["stream"]
        tokens[s] = tokens.get(s, 0) + sum(1 for l in e["labels"] if l != -100)
    total_tok = sum(tokens.values())
    print(f"\n{'stream':10s} {'examples':>9s} {'loss tokens':>13s} {'share':>7s}")
    for s, n in (("cpt", len(cpt)), ("sft", len(sft)), ("replay", len(replay))):
        print(f"{s:10s} {n:9d} {tokens.get(s, 0):13,d} {tokens.get(s, 0) / total_tok:6.1%}")
    print(f"{'total':10s} {len(examples):9d} {total_tok:13,d}")

    for s in ("cpt", "sft", "replay"):
        ex = next(e for e in examples if e["stream"] == s)
        show(tok, ex, s)
    print()

    model = AutoModelForCausalLM.from_pretrained(a.model)   # fp32; bf16=True autocasts
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    model.generation_config.eos_token_id = [tok.eos_token_id, eot]

    if a.lora:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(
            r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            # MLP modules included on purpose: knowledge injection needs them, and
            # attention-only LoRA is the usual reason a LoRA "won't learn facts".
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out,
        learning_rate=a.lr,
        num_train_epochs=50 if a.dry_run else a.epochs,
        max_steps=50 if a.dry_run else -1,
        per_device_train_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="no" if a.dry_run else "epoch",
        save_total_limit=None,          # keep every epoch: you need to score each one
        save_only_model=True,           # weights only: ~4 GB per epoch instead of ~12 GB
                                        # with optimizer state (so no resuming mid-run)
        bf16=True,
        seed=a.seed,
        report_to="none",
        remove_unused_columns=False,    # "stream"/"tag" are ours, not the model's
    )

    trainer = Trainer(model=model, args=args, train_dataset=MixedDataset(examples),
                      data_collator=Collator(tok.pad_token_id))
    trainer.train()

    if a.dry_run:
        print("\ndry run finished - nothing saved.")
        return

    trainer.save_model(a.out)
    tok.save_pretrained(a.out)          # persists the chat template for inference
    print(f"\nsaved to {a.out}")
    print("Now score EVERY epoch checkpoint on both evals:")
    print(f"  python score_eval_set.py  <checkpoint>   # cloze: did the book land (cheap, run every epoch)")
    print(f"  python eval_retention.py --model <checkpoint>   # what you kept")
    print(f"  python score_sft_eval.py  --model <checkpoint>   # chat answers, on the shortlist only")
    print("Take the last checkpoint before retention starts dropping, not the best book score.")


if __name__ == "__main__":
    main()