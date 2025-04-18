import torch
import wandb
import pickle
import datetime
import pandas as pd
import psycopg
from tqdm import tqdm
import model
import dataset
import config
import os

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

conn = psycopg.connect(
    "postgres://postgres:postgres@localhost:5432/hackernews")
query = "SELECT title, domain, timestamp, score FROM stories"
df = pd.read_sql(query, conn)
conn.close()

vocab_to_int = pickle.load(open(config.VOCAB_TO_ID_PATH, 'rb'))

ds = dataset.HackerNewsDataset(
    df['title'].values,
    df['domain'].values,
    df['timestamp'].values,
    df['score'].values,
    vocab_to_int,
    max_title_len=config.MAX_TITLE_LENGTH
)

train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

train_dl = torch.utils.data.DataLoader(
    train_ds, batch_size=config.REGRESSOR_BATCH_SIZE, shuffle=True)
val_dl = torch.utils.data.DataLoader(
    val_ds, batch_size=config.REGRESSOR_BATCH_SIZE)

skipgram = model.SkipGram(len(vocab_to_int), config.EMBEDDING_DIM)
skipgram.load_state_dict(torch.load(config.SKIPGRAM_BEST_MODEL_PATH))
skipgram.to(dev)
skipgram.eval()

regressor = model.Regressor(
    emb_dim=config.EMBEDDING_DIM, domain_size=len(ds.domain_to_idx))
regressor.to(dev)
optimizer = torch.optim.Adam(regressor.parameters(), lr=config.REGRESSOR_LR)
criterion = torch.nn.MSELoss()

wandb.init(project='mlx7-week1-regressor', name=f'{ts}')

best_val_loss = float('inf')

for epoch in range(config.REGRESSOR_EPOCHS):
    regressor.train()
    train_loss = 0
    train_progress = tqdm(
        train_dl, desc=f'Epoch {epoch+1} (Train)', leave=False)

    for batch in train_progress:
        title_ids = batch['title_ids'].to(dev)
        domain_idx = batch['domain_idx'].to(dev)
        time_feats = batch['time_feats'].to(dev)
        title_length = batch['title_length'].to(dev)
        score = batch['score'].to(dev)

        with torch.no_grad():
            title_emb = skipgram.emb(title_ids).mean(dim=1)

        optimizer.zero_grad()
        output = regressor(title_emb, domain_idx, time_feats, title_length)
        loss = criterion(output.squeeze(), score)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_progress.set_postfix({'loss': loss.item()})

    train_loss /= len(train_dl)

    regressor.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_dl, desc=f'Epoch {epoch+1} (Val)', leave=False):
            title_ids = batch['title_ids'].to(dev)
            domain_idx = batch['domain_idx'].to(dev)
            time_feats = batch['time_feats'].to(dev)
            title_length = batch['title_length'].to(dev)
            score = batch['score'].to(dev)

            title_emb = skipgram.emb(title_ids).mean(dim=1)
            output = regressor(title_emb, domain_idx, time_feats, title_length)
            loss = criterion(output.squeeze(), score)

            val_loss += loss.item()

    val_loss /= len(val_dl)

    wandb.log({
        'train_loss': train_loss,
        'val_loss': val_loss,
        'epoch': epoch
    })

    print(
        f'Epoch {epoch+1}: Train Loss = {train_loss:.4f}, '
        f'Val Loss = {val_loss:.4f}'
    )

    checkpoint_name = f'{epoch + 1}.pth'
    checkpoint_path = os.path.join(
        config.REGRESSOR_CHECKPOINT_DIR, checkpoint_name)
    torch.save(regressor.state_dict(), checkpoint_path)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(regressor.state_dict(), config.REGRESSOR_BEST_MODEL_PATH)
        torch.save(ds.domain_to_idx, config.DOMAIN_MAPPING_PATH)
        print(
            f'New best model saved at epoch {epoch+1} '
            f'with validation loss {best_val_loss:.4f}'
        )

print(f'Training completed! Best validation loss: {best_val_loss:.4f}')
wandb.finish()
