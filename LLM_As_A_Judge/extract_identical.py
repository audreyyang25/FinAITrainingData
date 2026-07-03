import pandas as pd
import json
from collections import defaultdict

df = pd.read_csv('runs/v3/gender_deltas.csv')
identical = df.loc[df['delta']==0, ['type', 'base_id', 'demographic', 'generator', 'judge']]
keys_list = set(identical[['type', 'base_id', 'demographic', 'generator', 'judge']].itertuples(index=False, name=None))

INPUT = 'runs/v3/judgments.jsonl'
OUTPUT = 'runs/v3/identical_score_judgments.jsonl'
grouped_data = defaultdict(list)

with open(INPUT, 'r', encoding='utf-8') as infile:
    for line in infile:
        line = line.strip()
        if not line:
            continue

        item = json.loads(line)

        item_key = (item.get('type'), item.get('base_id'), item.get('demographic'), item.get('generator'), item.get('judge'))

        if item_key in keys_list:
            grouped_data[item_key].append(item)
            
            
with open(OUTPUT, 'w', encoding='utf-8') as outfile:
    for item_key, items in grouped_data.items():
        for item in items:
            outfile.write(json.dumps(item) + '\n')