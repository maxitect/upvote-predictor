import collections
import pickle
import psycopg
import requests
import src.config as config


def preprocess(text):
    if not text:
        return []
    text = str(text).lower()
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace(':',  ' <COLON> ')
    words = text.split()
    return words


r = requests.get(
    'https://huggingface.co/datasets/ardMLX/text8/resolve/main/text8')
with open('text8', 'wb') as f:
    f.write(r.content)
with open('text8') as f:
    text8: str = f.read()

corpus: list[str] = preprocess(text8)
print(f"Corpus object type: {type(corpus)}")  # <class 'list'>
print(f"Original corpus size: {len(corpus)}")  # 17,005,207
# ['anarchism', 'originated', 'as', 'a', 'term', 'of', 'abuse']
print(f"First 7 words: {corpus[:7]}")

# once saved, check content with: head -c 100 corpus.json
with open(config.CORPUS_PATH, 'wb') as f:
    pickle.dump(corpus, f)

# Connect to the database
conn_string = "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki"
with psycopg.connect(conn_string) as conn:
    with conn.cursor() as cur:
        # Query only active comments
        cur.execute(
            """
            SELECT text FROM hacker_news.items
            WHERE type = 'comment' AND dead IS NULL AND time >= '2024-08-01';
            """
        )
        comments = [row[0] for row in cur.fetchall() if row[0]]

print(f"Retrieved {len(comments)} comments from Hacker News")

# Process comments and add to corpus
comment_words = []
for comment in comments:
    comment_words.extend(preprocess(comment))

# Filter out low frequency words
word_counts = collections.Counter(comment_words)
comment_words = [word for word in comment_words if word_counts[word] > 5]

print(f"Extracted {len(comment_words)} words from comments")

# Add to existing corpus
corpus.extend(comment_words)
print(f"New corpus size: {len(corpus)}")

# Save updated corpus
with open(config.CORPUS_PATH, 'wb') as f:
    pickle.dump(corpus, f)


# Recreate vocabulary lookup tables
def create_lookup_tables(words):
    word_counts = collections.Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab


words_to_ids, ids_to_words = create_lookup_tables(corpus)
print(f"Vocabulary size: {len(words_to_ids)}")  # 296,388
# Create tokens from updated corpus
tokens = [words_to_ids[word] for word in corpus]
print(f"Total tokens: {len(tokens)}")  # 61,026,684

print(type(tokens))  # <class 'list'>
print(tokens[:7])   # [10031, 5640, 23, 7, 421, 6, 2338]

print(ids_to_words[5234])        # threats
print(words_to_ids['anarchism'])  # 10,031
print(words_to_ids['have'])      # 33

# Save updated vocabulary
with open(config.VOCAB_TO_ID_PATH, 'wb') as f:
    pickle.dump(words_to_ids, f)
with open(config.ID_TO_VOCAB_PATH, 'wb') as f:
    pickle.dump(ids_to_words, f)
