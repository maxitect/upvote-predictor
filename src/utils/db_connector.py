import psycopg
import pandas as pd
import re


def clean_domain(url):
    if not url or pd.isna(url):
        return '<no_domain>'
    domain = re.search(r'(?:https?://)?(?:www\.)?([^/]+)', url)
    if domain:
        return domain.group(1).lower()
    return '<no_domain>'


def extract_hn_data():
    conn = psycopg.connect(
        "postgres://sy91dhb:g5t49ao@178.156.142.230:5432/hd64m1ki")

    query = """
    SELECT
        title,
        url,
        score,
        time,
        by as author
    FROM stories
    WHERE score >= 1
    AND title IS NOT NULL
    AND LENGTH(title) > 0
    ORDER BY time DESC
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df['domain'] = df['url'].apply(clean_domain)
    df['timestamp'] = pd.to_datetime(
        df['time'], unit='s').dt.strftime('%Y-%m-%dT%H:%M:%S')
    df['title_length'] = df['title'].apply(len)

    df = df[df['title_length'] <= 300]
    df = df[df['score'] <= 1000]

    return df[['title', 'domain', 'timestamp', 'score', 'author']]


def load_to_new_db(df):
    conn = psycopg.connect(
        "postgres://postgres:postgres@localhost:5432/hackernews")

    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS stories (
        id SERIAL PRIMARY KEY,
        title TEXT NOT NULL,
        domain TEXT,
        timestamp TIMESTAMP,
        score INTEGER,
        author TEXT
    )
    """)

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO stories (title, domain, timestamp, score, author)
            VALUES (%s, %s, %s, %s, %s)""",
                       (row['title'], row['domain'],
                        row['timestamp'], row['score'], row['author'])
                       )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    df = extract_hn_data()
    print(f"Extracted {len(df)} records")
    print(df.head())
    load_to_new_db(df)
    print("Data loaded to new database")
