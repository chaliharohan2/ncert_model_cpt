from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import json
import sys
import torch

# model_name = "google/gemma-3-1b-it"
# model_name = "google/gemma-3-1b-pt"
# model_name = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v8"
model_name = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_sft/models/gemma_3_1B_ncert_sft_v1"

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
stream = True

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

# for our sft model
# identical to the template in train_sft.py
if "ncert" in model_name:
    print("Loading sft tokenizer settings....\n")
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

    # should not go to this since chat template is already saved
    if tokenizer.chat_template is None:
        tokenizer.chat_template = CHAT_TEMPLATE

    stop_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    
#####

# load model
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name, dtype=torch.bfloat16)
model.to(device).eval()

if __name__ == "__main__":

    while True:
        user_input = input("> ").strip()
        if user_input == "q":
            print("\nExiting....")
            sys.exit(0)
        elif not user_input:
            continue

        tokens = tokenizer.apply_chat_template([{"role": "user", "content": user_input}], add_generation_prompt=True,
                                          tokenize=True, return_tensors="pt", return_dict=True).to(device)

        with torch.inference_mode():
            # streaming approach
            streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=False, skip_special_tokens=True)
            output = model.generate(**tokens, max_new_tokens=200, streamer=streamer, do_sample=False,
                           repetition_penalty=1.0, eos_token_id=stop_ids,
                           pad_token_id=tokenizer.pad_token_id,)
            # resp = tokenizer.decode(output, skip_special_tokens=False)[0]
            # print(resp)
