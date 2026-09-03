import json

def load_dataset(path: str):
    dataset = []
    with open(path, mode="r") as f:
        for line in f.readlines():
            dataset.append(json.loads(s=line)["text"])
    return dataset

if __name__ == "__main__":
    load_dataset("/home/nz-dgx-spark-01/Documents/Nyalazone/ncert_model_content_training/dataset/cpt_pool.jsonl")