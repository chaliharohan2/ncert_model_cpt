"""
Interactive chat with the SFT model, using the exact prompt format it was trained on.

Why an earlier session echoed the question and kept generating new Q&A pairs:
if the question is sent as raw text, the model continues it the way it learned
during CPT ("Q: ... A: ... Q: ..."). SFT taught a different route -
<start_of_turn>user ... <start_of_turn>model ... - and the model only stops
cleanly when it is prompted that way AND <end_of_turn> is a stop token.
TextStreamer also prints the prompt unless skip_prompt=True, which is where
the echoed question came from.

Usage
  python chat.py                 # correct chat format
  python chat.py --show-prompt   # also print the exact formatted prompt
  python chat.py --raw           # reproduce the old behaviour (raw text, no template)
  python chat.py --rp 1.15       # add a repetition penalty (diagnostic)

Each question is independent: SFT used single-turn examples only, so the model
has never seen a conversation history. Don't feed previous turns back in.
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

MODEL_DIR = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_sft/models/gemma_3_1B_ncert_sft_v1"

# identical to the template in train_sft.py (only used if the saved tokenizer lacks one)
CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
    "{% if message['role'] == 'user' %}"
    "<start_of_turn>user\n{{ message['content'] | trim }}<end_of_turn>\n"
    "{% elif message['role'] == 'assistant' %}"
    "<start_of_turn>model\n"
    "{% generation %}{{ message['content'] | trim }}<end_of_turn>{% endgeneration %}\n"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DIR)
    ap.add_argument("--raw", action="store_true", help="send raw text with no chat template (the old, wrong way)")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--rp", type=float, default=1.0, help="repetition penalty; 1.0 = off")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.chat_template is None:
        tok.chat_template = CHAT_TEMPLATE
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(device).eval()

    stop_ids = [tok.eos_token_id, tok.convert_tokens_to_ids("<end_of_turn>")]
    # skip_prompt=True: print only the model's reply, not the question back to you
    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)

    print(f"Loaded {a.model}")
    print("Mode:", "RAW TEXT (old behaviour)" if a.raw else "chat template")
    print("Ask a question, or q to quit. Each question is answered independently.")

    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"q", "quit", "exit"}:
            break
        if not q:
            continue

        if a.raw:
            enc = tok(q, return_tensors="pt").to(device)
        else:
            enc = tok.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True,
                                          tokenize=True, return_tensors="pt", return_dict=True).to(device)
        if a.show_prompt:
            print("[prompt]", repr(tok.decode(enc["input_ids"][0])))

        with torch.inference_mode():
            model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False,
                           repetition_penalty=a.rp, eos_token_id=stop_ids,
                           pad_token_id=tok.pad_token_id, streamer=streamer)


if __name__ == "__main__":
    main()