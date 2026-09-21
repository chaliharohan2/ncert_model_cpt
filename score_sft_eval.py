"""
Generation-based SFT evaluation for the NCERT study model.

Runs every model in MODELS over sft_eval_v2.jsonl in chat format, then scores:
  Tier A  factual_qa / binding_pair / paraphrase_set  - first-sentence string match
          unanswerable                                - refused / deflected / fabricated
          out_of_scope                                - informational only
  Tier B  summary / passage_locate / question_bank / personal_reflection
          - exported with checklists for judging, plus cheap automatic proxies
  All     clean_stop, runaway, verbatim copying, length, false_deflection

Usage
  python score_sft_eval.py                  # generate + score every model in MODELS
  python score_sft_eval.py --models sft_v1  # just one model
  python score_sft_eval.py --reuse          # re-score existing generations, no GPU work
  python score_sft_eval.py --selftest       # score the REFERENCE answers (no model) to
                                            # check the scorer; Tier A should be ~100%
"""
import argparse, json, os, re, csv
from collections import defaultdict, Counter

# ---------------------------------------------------------------- config
ROOT = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training"
EVAL_PATH = f"{ROOT}/dataset/sft_eval_v2.jsonl"
CORPUS_PATH = f"{ROOT}/dataset/cpt_pool.jsonl"
OUT_DIR = f"{ROOT}/eval_results/sft_eval"

# force_sft_template=True -> use the exact template SFT was trained with.
# Use it for local checkpoints; keep the hub instruct model on its own template.
MODELS = {
    "sft_v1":        {"path": f"{ROOT}/ncert_model_sft/models/gemma_3_1B_ncert_sft_v1", "force_sft_template": True},
    "gemma-3-1b-it": {"path": "google/gemma-3-1b-it",                                   "force_sft_template": False},
    "cpt_v8":        {"path": f"{ROOT}/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v8", "force_sft_template": True},
}

MAX_NEW_TOKENS = 256
BATCH_SIZE = 8
VERBATIM_N = 15   # a 15-word run shared with the book counts as copying

# identical to the template in train_sft.py
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

ANSWERABLE = {"factual_qa", "binding_pair", "paraphrase_set"}
TIER_B = {"summary", "passage_locate", "question_bank", "personal_reflection"}
PERSONAL_CHECKLIST = [
    "does NOT invent a personal experience as its own ('I once...')",
    "gives the student something concrete (a prompt, an angle, an example), not only a description of the exercise",
    "connects back to the chapter",
]

# ---------------------------------------------------------------- text utilities
# Do not end a sentence on an initial (R. K. Laxman), a title (Mr.), or B.C./A.D.
_ABBR = re.compile(r"(?:\b(?:Mr|Mrs|Ms|Dr|St|Jr|Sr|vs|etc|No)|\b[A-Z]|\bB\.C|\bA\.D|e\.g|i\.e)\.$")

def first_sentence(text):
    text = (text or "").strip()
    for m in re.finditer(r"[.!?](?=\s|$)", text):
        chunk = text[: m.end()]
        if _ABBR.search(chunk):
            continue
        return chunk
    return text

def term_pos(term, text):
    """Position of a whole-word match, or None. '25' must not match inside '125';
    '100' not inside '100,000'; 'ten' not inside 'often'."""
    t = re.escape(term.lower().strip())
    m = re.search(r"(?<![a-z0-9])" + t + r"(?![a-z0-9]|,\d)", text.lower())
    return m.start() if m else None

def has_term(term, text):
    return term_pos(term, text) is not None

def first_value_pass(item, first):
    """Pass if every required group is present in the first sentence AND the model's
    FIRST-STATED value is a required one. So 'He asked for a hundred but got seventy'
    passes the 'how much did he ask for' item (it leads with the right value), while
    'He got seventy, though he asked for a hundred' fails (it leads with the wrong one).
    Plain 'no forbidden term in the first sentence' would fail the first answer, which
    is exactly the contrastive style SFT was trained to produce."""
    req_pos = []
    for grp in item.get("required_any", []):
        hits = [p for p in (term_pos(a, first) for a in grp) if p is not None]
        if not hits:
            return False, []
        req_pos.append(min(hits))
    forb = [(term_pos(f, first), f) for f in item.get("forbidden", [])]
    forb = [(p, f) for p, f in forb if p is not None]
    if forb and req_pos and min(p for p, _ in forb) < min(req_pos):
        return False, [f for _, f in forb]
    return True, [f for _, f in forb]

# Broad refusal detector, used IN ADDITION to each item's own list. The item lists were
# too narrow: half the reference answers ("never explains", "is never given") failed them.
_V = r"(?:say|name|mention|tell|give|state|describe|explain|specify|reveal|identify|record|provide)"
_VED = r"(?:said|says|named|names|mentioned|mentions|told|tells|given|gives|stated|states|described|describes|explained|explains|specified|specifies|revealed|reveals|identified|identifies|recorded|records)"
REFUSAL_RE = re.compile("|".join([
    r"\b(?:doesn't|does not|didn't|did not|don't|do not)\s+" + _V + r"\b",
    r"\bnever\s+" + _VED + r"\b",
    r"\b(?:is|are|was|were)\s+(?:not|never)\s+" + _VED + r"\b",
    r"\b(?:isn't|aren't|wasn't|weren't)\s+" + _VED + r"\b",
    r"\bno\s+(?:name|mention|breed|date|description|details?|explanation|information|indication|record)\b[^.]{0,40}\b(?:given|mentioned|provided|stated|offered|named)\b",
    r"\bthe\s+(?:book|text|story|chapter|poem|play|essay|lesson|passage)\s+(?:doesn't|does not|never)\b",
    r"\b(?:unnamed|unspecified|not known)\b",
]), re.I)

# Matches the pattern used to find the deflection records in sft_set.jsonl.
DEFLECTION_RE = re.compile(
    r"(this is an? (?:open|personal|opinion)|no single (?:correct|right|one) answer|no one (?:correct|right) answer"
    r"|open[- ]ended|students (?:are|were|should|could) (?:meant|asked|expected|encouraged|invited)"
    r"|invit\w+ students|answers? will vary|opinion[- ]based|personal[- ]response)", re.I)

def is_refusal(first, item):
    if REFUSAL_RE.search(first):
        return True
    return any(has_term(a, first) for g in item.get("required_any", []) for a in g)

_COMMON = set("""The He She It His Her Hers Him They Them Their There This That These Those
No Yes In On At As An A And But Or So If When Where What Which Who Why How Only Also Then
Its Our We You I My Your Both Each Every Some Any One Two Three Four Five Six Seven Eight
Nine Ten First Second Third Here However Instead Although While Because After Before""".split())

def out_of_book_entities(answer, corpus_lower, prompt):
    """Capitalised names / numbers in the answer that appear nowhere in the book or prompt.
    Not proof of fabrication (general knowledge can be correct) but a strong signal."""
    ctx = corpus_lower + " " + (prompt or "").lower()
    names = {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", answer or "") if w not in _COMMON and w.lower() not in ctx}
    nums = {n for n in re.findall(r"\b\d[\d,]*\b", answer or "") if n not in ctx}
    return sorted(names | nums)

def words(s):
    return re.findall(r"[a-z0-9']+", (s or "").lower())

def build_ngrams(texts, n):
    grams = set()
    for t in texts:
        w = words(t)
        for i in range(len(w) - n + 1):
            grams.add(tuple(w[i : i + n]))
    return grams

def longest_verbatim(answer, grams, n):
    w, best, i = words(answer), 0, 0
    while i <= len(w) - n:
        if tuple(w[i : i + n]) in grams:
            j = i + n
            while j < len(w) and tuple(w[j - n + 1 : j + 1]) in grams:
                j += 1
            best, i = max(best, j - i), j
        else:
            i += 1
    return best

def is_runaway(raw, hit_limit):
    return hit_limit or "<start_of_turn>" in raw or re.search(r"(^|\n)\s*(Q|Question)\s*[:.]", raw) is not None

# ---------------------------------------------------------------- generation
def generate(name, cfg, items):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["path"])
    if cfg["force_sft_template"] or tok.chat_template is None:
        tok.chat_template = CHAT_TEMPLATE
    tok.padding_side = "left"   # decoder-only batched generation must pad on the left
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    stop_ids = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tok.unk_token_id:
        stop_ids.append(eot)

    model = AutoModelForCausalLM.from_pretrained(cfg["path"], dtype=torch.bfloat16).to("cuda").eval()

    # one-time sanity check on prompt formatting
    probe = tok.apply_chat_template([{"role": "user", "content": "hi"}], add_generation_prompt=True, tokenize=True)
    ids = probe if isinstance(probe, list) else probe["input_ids"]
    print(f"[{name}] prompt format: {tok.decode(ids)!r}  (BOS count = {ids.count(tok.bos_token_id)}, want 1)")

    out = []
    for b in range(0, len(items), BATCH_SIZE):
        batch = items[b : b + BATCH_SIZE]
        convs = [[{"role": "user", "content": x["prompt"]}] for x in batch]
        enc = tok.apply_chat_template(convs, add_generation_prompt=True, tokenize=True, padding=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                 eos_token_id=stop_ids, pad_token_id=tok.pad_token_id)
        new = gen[:, enc["input_ids"].shape[1]:]
        for x, row in zip(batch, new.tolist()):
            stop_at = next((i for i, t in enumerate(row) if t in stop_ids), None)
            kept = row if stop_at is None else row[:stop_at]
            out.append({
                "eval_id": x["eval_id"],
                "answer": tok.decode(kept, skip_special_tokens=True).strip(),
                "raw": tok.decode(kept, skip_special_tokens=False),
                "clean_stop": stop_at is not None,
                "hit_limit": stop_at is None,
            })
        print(f"[{name}] {min(b + BATCH_SIZE, len(items))}/{len(items)}")

    del model
    torch.cuda.empty_cache()
    return out

# ---------------------------------------------------------------- scoring
def score(items, gens, corpus_lower, grams):
    g = {x["eval_id"]: x for x in gens}
    rows = []
    for it in items:
        gen = g[it["eval_id"]]
        ans, cat = gen["answer"], it["category"]
        first = first_sentence(ans)
        r = {
            "eval_id": it["eval_id"], "category": cat, "tier": it["tier"], "prompt": it["prompt"],
            "answer": ans, "first_sentence": first,
            "clean_stop": gen["clean_stop"], "runaway": is_runaway(gen["raw"], gen["hit_limit"]),
            "verbatim_words": longest_verbatim(ans, grams, VERBATIM_N),
            "length_words": len(ans.split()),
            "deflection": bool(DEFLECTION_RE.search(ans)),
            "out_of_book": out_of_book_entities(ans, corpus_lower, it["prompt"]),
        }
        if cat in ANSWERABLE:
            r["pass"], r["forbidden_mentioned"] = first_value_pass(it, first)
            r["group_id"] = it.get("group_id")
            r["paraphrase_group"] = it.get("paraphrase_group")
        elif cat == "unanswerable":
            if is_refusal(first, it):
                r["outcome"] = "refused"
            elif r["deflection"]:
                r["outcome"] = "deflected"
            else:
                r["outcome"] = "fabricated"
            r["refused_but_speculated"] = r["outcome"] == "refused" and bool(r["out_of_book"])
        rows.append(r)
    return rows

def summarize(rows, items):
    by_id = {x["eval_id"]: x for x in items}
    s = {}
    for cat in ["factual_qa", "binding_pair", "paraphrase_set"]:
        rr = [r for r in rows if r["category"] == cat]
        s[cat] = 100 * sum(r["pass"] for r in rr) / max(1, len(rr))

    grp = defaultdict(list)
    for r in rows:
        if r["category"] == "binding_pair":
            grp[r["group_id"]].append(r["pass"])
    s["binding_groups_all_pass"] = 100 * sum(all(v) for v in grp.values()) / max(1, len(grp))
    s["binding_groups_failed"] = sorted(k for k, v in grp.items() if not all(v))

    pg = defaultdict(list)
    for r in rows:
        if r["category"] == "paraphrase_set":
            pg[r["paraphrase_group"]].append(r["pass"])
    s["paraphrase_all_three"] = 100 * sum(all(v) for v in pg.values()) / max(1, len(pg))

    un = [r for r in rows if r["category"] == "unanswerable"]
    oc = Counter(r["outcome"] for r in un)
    for k in ["refused", "deflected", "fabricated"]:
        s[f"unans_{k}"] = 100 * oc[k] / max(1, len(un))
    s["unans_refused_but_speculated"] = sum(r["refused_but_speculated"] for r in un)

    ans_rows = [r for r in rows if r["category"] in ANSWERABLE]
    s["false_deflection"] = 100 * sum(r["deflection"] for r in ans_rows) / max(1, len(ans_rows))
    s["clean_stop"] = 100 * sum(r["clean_stop"] for r in rows) / len(rows)
    s["runaway"] = 100 * sum(r["runaway"] for r in rows) / len(rows)
    s["verbatim_ge15"] = sum(r["verbatim_words"] >= VERBATIM_N for r in rows)
    s["median_words"] = sorted(r["length_words"] for r in rows)[len(rows) // 2]
    return s

def export_tier_b(name, rows, items, out_dir):
    by_id = {x["eval_id"]: x for x in items}
    path = os.path.join(out_dir, f"tierB_{name}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["eval_id", "category", "prompt", "model_answer", "checklist", "auto_checks", "reference", "score_0_to_1", "notes"])
        for r in rows:
            if r["category"] not in TIER_B:
                continue
            it = by_id[r["eval_id"]]
            cat, ans = r["category"], r["answer"]
            auto = []
            if cat == "summary":
                checklist = [f"covers: {c}" for c in it.get("must_cover", [])] + [f"does NOT include: {c}" for c in it.get("must_not_include", [])]
                mw = it.get("max_words")
                if mw:
                    auto.append(f"words={r['length_words']}/{mw} {'OK' if r['length_words'] <= mw else 'OVER'}")
                leaked = [c for c in it.get("must_not_include", []) if c.lower() in ans.lower()]
                if leaked:
                    auto.append(f"cross-chapter terms found: {leaked}")
            elif cat == "passage_locate":
                checklist = [f"location: {it.get('expected_location')}"] + [f"covers: {c}" for c in it.get("must_cover", [])]
                auto.append(f"verbatim_run={r['verbatim_words']} {'FAIL' if r['verbatim_words'] >= VERBATIM_N else 'OK'}")
            elif cat == "question_bank":
                n_q = len([ln for ln in ans.splitlines() if ln.strip().endswith("?")])
                checklist = [f"{it.get('required_count')} questions", f"answerable from {it.get('must_be_answerable_from')}"]
                auto.append(f"questions_found={n_q}/{it.get('required_count')}")
            else:  # personal_reflection
                checklist = PERSONAL_CHECKLIST
                if re.search(r"\bI (?:once|remember|have been|was|had|felt|went|used to)\b", ans):
                    auto.append("POSSIBLE invented personal experience ('I once/I remember...')")
            if r["runaway"]:
                auto.append("RUNAWAY")
            w.writerow([r["eval_id"], cat, r["prompt"], ans, " | ".join(checklist), " | ".join(auto), it.get("reference") or "", "", ""])
    return path

def print_table(summaries):
    names = list(summaries)
    line = lambda label, key, fmt="{:>13.1f}%": print(f"{label:<30}" + "".join(fmt.format(summaries[n][key]) for n in names))
    print("\n" + "=" * (30 + 14 * len(names)))
    print(f"{'metric':<30}" + "".join(f"{n:>14}" for n in names))
    print("-" * (30 + 14 * len(names)))
    print("ANSWERING")
    line("  factual_qa", "factual_qa")
    line("  binding_pair (per item)", "binding_pair")
    line("  binding GROUPS all-pass", "binding_groups_all_pass")
    line("  paraphrase (per item)", "paraphrase_set")
    line("  paraphrase sets all-3", "paraphrase_all_three")
    print("UNANSWERABLE (sums to 100)")
    line("  refused", "unans_refused")
    line("  deflected", "unans_deflected")
    line("  fabricated", "unans_fabricated")
    line("  refused-but-speculated (n)", "unans_refused_but_speculated", "{:>14d}")
    print("BEHAVIOUR")
    line("  clean_stop", "clean_stop")
    line("  runaway", "runaway")
    line("  false_deflection", "false_deflection")
    line("  verbatim >=15 words (n)", "verbatim_ge15", "{:>14d}")
    line("  median answer words", "median_words", "{:>14d}")
    print("=" * (30 + 14 * len(names)))
    for n in names:
        if summaries[n]["binding_groups_failed"]:
            print(f"{n} failed binding groups: {', '.join(summaries[n]['binding_groups_failed'])}")

# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--eval", default=EVAL_PATH)
    ap.add_argument("--corpus", default=CORPUS_PATH)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--reuse", action="store_true", help="skip generation if a generations file exists")
    ap.add_argument("--selftest", action="store_true", help="score the reference answers instead of a model")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    items = [json.loads(l) for l in open(a.eval) if l.strip()]
    pool = [json.loads(l) for l in open(a.corpus) if l.strip()]
    book = [r["text"] for r in pool if r["type"] == "raw_text"]
    corpus_lower = ("\n".join(book) + "\n" + "\n".join(r["text"] for r in pool if r["type"] == "vocab_fact")).lower()
    grams = build_ngrams(book, VERBATIM_N)

    summaries = {}
    targets = ["REFERENCE"] if a.selftest else a.models
    for name in targets:
        gpath = os.path.join(a.out, f"generations_{name}.jsonl")
        if a.selftest:
            gens = [{"eval_id": x["eval_id"], "answer": x.get("reference") or "", "raw": x.get("reference") or "",
                     "clean_stop": True, "hit_limit": False} for x in items]
        elif a.reuse and os.path.exists(gpath):
            gens = [json.loads(l) for l in open(gpath)]
            print(f"[{name}] reusing {gpath}")
        else:
            gens = generate(name, MODELS[name], items)
            with open(gpath, "w") as f:
                for g in gens:
                    f.write(json.dumps(g, ensure_ascii=False) + "\n")

        rows = score(items, gens, corpus_lower, grams)
        with open(os.path.join(a.out, f"scored_{name}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tb = export_tier_b(name, rows, items, a.out)
        summaries[name] = summarize(rows, items)
        print(f"[{name}] per-item scores -> scored_{name}.jsonl ; Tier B for judging -> {os.path.basename(tb)}")

    print_table(summaries)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summaries, f, indent=2)

if __name__ == "__main__":
    main()