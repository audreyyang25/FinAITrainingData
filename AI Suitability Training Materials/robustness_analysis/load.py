import json, os, glob
import pandas as pd

DATA_DIR = "/Users/audreyyang/FinAITrainingData/AI Suitability Training Materials"

def detect_type(path: str) -> str:
    if path.lower().endswith(("standard.json", "p12.json")):
        return("standard")
    elif path.lower().endswith(("borderline.json", "p13.json")):
        return("borderline")
    elif path.lower().endswith("p14.json"):
        return("conversation")
    elif path.lower().endswith("p15.json"):
        return("red_flag")
    elif path.lower().endswith("p16.json"):
        return("adversarial")
    else:
        raise ValueError("Invalid classification.")

def load_all(data_dir: str = DATA_DIR) -> pd.DataFrame:
    rows = []
    for path in glob.glob(os.path.join(data_dir, "**", "*.json"), recursive=True):
        content_type = detect_type(path)
        records = json.load(open(path))
        for r in records:
            r["content_type"] = content_type
            r["_file"] = os.path.relpath(path, data_dir)
            rows.append(r)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = load_all()
    print(len(df), "records")
    print(df["content_type"].value_counts())
    print(df.columns.tolist())