from src.utils.db_connector import extract_hn_data, load_to_new_db

print("Starting data processing...")

# Extract data from remote database
df = extract_hn_data()
print(f"Processed {len(df)} records")
print("Sample data:")
print(df.head())

# Load data to local database
load_to_new_db(df)

print("\nStatistics of processed data:")
print(f"Total records: {len(df)}")
print(f"Average score: {df['score'].mean():.2f}")
print(f"Max score: {df['score'].max()}")
print(f"Average title length: {df['title_length'].mean():.2f} characters")
print("Top 5 domains by average score:")
print(df.groupby('domain')['score'].mean(
).sort_values(ascending=False).head(5))

print("\nWeekday score distribution:")
day_names = ['Monday', 'Tuesday', 'Wednesday',
             'Thursday', 'Friday', 'Saturday', 'Sunday']
day_stats = df.groupby('day_of_week')['score'].mean().reindex(range(7))
for day in range(7):
    print(f"{day_names[day]}: {day_stats[day]:.2f}")

print("\nData processing complete!")
