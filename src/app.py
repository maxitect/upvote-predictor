import os
import json
import torch
import pickle
import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import model

model_version = "0.1.0"
log_dir_path = "/var/log/app"
log_path = f"{log_dir_path}/V-{model_version}.log"

os.makedirs(log_dir_path, exist_ok=True)
if not os.path.exists(log_path):
    with open(log_path, 'w') as f:
        f.write('')

app = FastAPI()


class Post(BaseModel):
    author: str
    title: str
    timestamp: str


vocab_to_int = pickle.load(
    open('./models/skipgram/tkn_words_to_ids.pkl', 'rb'))
skipgram = model.SkipGram(len(vocab_to_int), 128)
skipgram.load_state_dict(torch.load(
    './models/skipgram/best_model.pth'))
skipgram.eval()

regressor = model.Regressor(emb_dim=128, domain_size=1000)
regressor.load_state_dict(torch.load(
    './models/regressor/best_model.pth'))
regressor.eval()

domain_to_idx = torch.load('./models/domain/domain_to_idx.pth')


def extract_domain(title):
    words = title.lower().split()
    for word in words:
        if '.' in word:
            return word
    return '<no_domain>'


def preprocess_title(title, max_len=100):
    title_tokens = title.lower().split()
    title_ids = []
    for word in title_tokens[:max_len]:
        if word in vocab_to_int:
            title_ids.append(vocab_to_int[word])
        else:
            title_ids.append(0)

    if len(title_ids) < max_len:
        title_ids.extend([0] * (max_len - len(title_ids)))

    return torch.tensor([title_ids])


def get_time_features(timestamp):
    dt = datetime.datetime.fromisoformat(timestamp)
    day_of_week = dt.weekday()
    hour_of_day = dt.hour
    return torch.tensor([[day_of_week/6, hour_of_day/23]], dtype=torch.float32)


@app.get("/ping")
def ping():
    return "ok"


@app.get("/version")
def version():
    return {"version": model_version}


@app.get("/logs")
def logs():
    with open(log_path, 'r') as f:
        log_entries = f.readlines()

    parsed_logs = []
    for entry in log_entries:
        if entry.strip():
            try:
                parsed_logs.append(json.loads(entry))
            except json.JSONDecodeError:
                continue

    return {"logs": parsed_logs}


@app.post("/how_many_upvotes")
async def how_many_upvotes(post: Post):
    start_time = datetime.datetime.now().timestamp()

    try:
        title_ids = preprocess_title(post.title)
        domain = extract_domain(post.title)
        domain_idx = torch.tensor([domain_to_idx.get(domain, 0)])
        time_feats = get_time_features(post.timestamp)
        title_length = torch.tensor(
            [min(len(post.title), 200)/200], dtype=torch.float32)

        with torch.no_grad():
            title_emb = skipgram.emb(title_ids).mean(dim=1)
            prediction = regressor(title_emb, domain_idx,
                                   time_feats, title_length)

        upvotes = max(1, int(prediction.item()))

        end_time = datetime.datetime.now().timestamp()
        latency = (end_time - start_time) * 1000

        log_entry = {
            "Latency": latency,
            "Version": model_version,
            "Timestamp": end_time,
            "Input": post.dict(),
            "Prediction": upvotes
        }

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        return {"upvotes": upvotes}

    except Exception as e:
        end_time = datetime.datetime.now().timestamp()
        latency = (end_time - start_time) * 1000

        log_entry = {
            "Latency": latency,
            "Version": model_version,
            "Timestamp": end_time,
            "Input": post.model_dump(),
            "Error": str(e)
        }

        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        raise HTTPException(status_code=500, detail=str(e))
