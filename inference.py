from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import json
import sys
import torch

# model_name = "google/gemma-3-1b-it"
# model_name = "google/gemma-3-1b-pt"
model_name = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/gemma_3_1B_pt_ncert_cpt_v2"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
stream = True

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
model.to(device)

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

if __name__ == "__main__":

    while True:
        user_input = input("> ").strip()
        if user_input == "q":
            print("\nExiting....")
            sys.exit(0)
        elif not user_input:
            continue

        tokens = tokenizer(user_input, return_tensors="pt").to(device)

        with torch.no_grad():
            # streaming approach
            streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=False, skip_special_tokens=True)
            output = model.generate(**tokens, max_new_tokens=200, streamer=streamer)
            # resp = tokenizer.decode(output, skip_special_tokens=False)[0]
            # print(resp)
