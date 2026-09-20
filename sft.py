from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
import torch
import json
import sys
from datasets import Dataset, load_dataset

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
MAX_LENGTH = 8000
DATASET_PATH = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/dataset/cpt_pool.jsonl"
MODEL_SAVE_PATH = "/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/ncert_model_cpt/models/qwen_3_1point7B_base_ncert_cpt_v3"
def tokenize_fn(batch, tokenizer: AutoTokenizer):
    return tokenizer(
        batch["text"],
        truncation=False,
        max_length=MAX_LENGTH
    )

model_name = "Qwen/Qwen3-1.7B-Base"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=model_name)
model.to(DEVICE)

if __name__ == "__main__":
    # load the dataset
    dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

    training_args = SFTConfig(
        output_dir=MODEL_SAVE_PATH,
        learning_rate=1.0e-5,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        seed=64,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        logging_steps=100,
        assistant_only_loss=True,
        bf16=True,
        max_length=8000
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        processing_class=tokenizer
    )

    trainer.train()
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)