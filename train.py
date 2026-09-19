from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import torch
import json
import sys
from prepare_dataset import load_dataset
from datasets import Dataset

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
    texts = load_dataset(path=DATASET_PATH)
    dataset = Dataset.from_dict({"text": texts})
    final_dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        fn_kwargs={"tokenizer": tokenizer}
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    """
    # Model = gemma 3 1B base model (google/gemma-3-1b-pt)

    v1 and v2 were 3 epochs, lr = 1.0e-5
    v3 is 8 epochs, lr = 1.0e-5
    v4 is 15 epochs, lr = 1.0e-5
    v5 is 25 epochs, lr = 1.0e-5
    v6 is 8 epochs, lr = 5.0e-5
    v7 is 8 epochs, lr = 1.0e-4
    v8 is 15 epochs, lr = 5.0e-5
    v9 is 15 epochs, lr = 1.0e-4

    # Model = Qwen 3 1.7 B Base model (Qwen/Qwen3-1.7B-Base)

    v1 is 15 epochs, lr = 3.0e-5
    v2 is 15 epochs, lr = 5.0e-5
    v3 is 8 epochs, lr = 1.0e-4
    """
    training_args = TrainingArguments(
        output_dir=MODEL_SAVE_PATH,
        learning_rate=1.0e-4,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        num_train_epochs=8,
        seed=64,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,
        logging_steps=100,
        bf16=True
    )

    trainer = Trainer(
        model=model,
        train_dataset=final_dataset,
        data_collator=data_collator,
        args=training_args
    )

    trainer.train()
    trainer.save_model(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)