import requests
import pandas as pd
import os
import pickle
import src.config as config

# Create directories
os.makedirs(config.DOMAIN_CHECKPOINT_DIR, exist_ok=True)

print("Downloading Hacker News dataset...")
# Download the dataset as a parquet file
r = requests.get('https://huggingface.co/datasets/artemisweb/hackernewsupvotes/resolve/main/data/train.parquet')
with open('hn_data.parquet', 'wb') as f:
    f.write(r.content)

# Read the parquet file with pandas
print("Processing dataset...")
df = pd.read_parquet('hn_data.parquet')

# Extract the needed columns
titles = df['title'].fillna('').tolist()
urls = df['url'].fillna('').tolist()
domains = [url.split('/')[2] if url and '/' in url else '<no_domain>' for url in urls]
timestamps = [pd.to_datetime(time).isoformat() for time in df['time']]
scores = df['score'].tolist()

print(f"Processed {len(titles)} records")

# Create domain mapping
domain_to_idx = {d: i+1 for i, d in enumerate(set(domains))}
domain_to_idx['<UNK>'] = 0

# Save processed data
hn_dataset = {
    'titles': titles,
    'domains': domains,
    'timestamps': timestamps, 
    'scores': scores
}

print("Saving data...")
with open(os.path.join(config.DOMAIN_CHECKPOINT_DIR, 'hn_dataset.pkl'), 'wb') as f:
    pickle.dump(hn_dataset, f)

with open(config.DOMAIN_MAPPING_PATH, 'wb') as f:
    pickle.dump(domain_to_idx, f)

print(f"Data processing complete: {len(titles)} records saved")