import torch
import wandb
import pickle
import datetime
import pandas as pd
import psycopg
from tqdm import tqdm
import model
import dataset

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ts = datetime.datetime.now().strftime('%Y_%m_%d__%H_%M_%S')

conn = psycopg.connect(
    "postgres://postgres:postgres@localhost:5432/hackernews")
query = "SELECT title, domain, timestamp, score FROM stories"
df = pd.read_sql(query, conn)
conn.close()

vocab_to_int = pickle.load(open('./tkn_words_to_ids.pkl', 'rb'))
int_to_vocab = pickle.load(open('./tkn_ids_to_words.pkl', 'rb'))

ds = dataset.HackerNewsDataset(
    df['title'].values,
    df['domain'].values,
    df['timestamp'].values,
    df['score'].values,
    vocab_to_int
)

train_size = int(0.8 * len(ds))
val_size = len(ds) - train_size
train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=64)

skipgram = model.SkipGram(len(vocab_to_int), 128)
skipgram.load_state_dict(torch.load(
    './checkpoints/2025_04_17__11_04_09.5.skipgram.pth'))
skipgram.to(dev)
skipgram.eval()

regressor = model.Regressor(emb_dim=128, domain_size=len(ds.domain_to_idx))
regressor.to(dev)
optimizer = torch.optim.Adam(regressor.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

wandb.init(project='mlx7-week1-regressor', name=f'{ts}')

for epoch in range(10):
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

    if (epoch + 1) % 5 == 0:
        checkpoint_name = f'{ts}.{epoch + 1}.regressor.pth'
        torch.save(regressor.state_dict(), f'./checkpoints/{checkpoint_name}')
        artifact = wandb.Artifact('regressor-weights', type='model')
        artifact.add_file(f'./checkpoints/{checkpoint_name}')
        wandb.log_artifact(artifact)

final_checkpoint = f'{ts}.final.regressor.pth'
torch.save(regressor.state_dict(), f'./checkpoints/{final_checkpoint}')
torch.save(ds.domain_to_idx, './checkpoints/domain_to_idx.pth')

print('Training completed!')
wandb.finish()
