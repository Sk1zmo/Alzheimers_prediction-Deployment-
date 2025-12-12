import sqlite3
from datetime import datetime

# Create or open local DB file
conn = sqlite3.connect("alzheimers.db", check_same_thread=False)
cursor = conn.cursor()

# Create the table if missing
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    filename TEXT,
    image_url TEXT,
    predicted_class TEXT,
    confidence REAL
)
""")
conn.commit()


def save_prediction(filename, image_url, predicted_class, confidence):
    timestamp = datetime.utcnow().isoformat()

    cursor.execute("""
        INSERT INTO predictions (timestamp, filename, image_url, predicted_class, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (timestamp, filename, image_url, predicted_class, confidence))
    
    conn.commit()


def get_all_predictions():
    cursor.execute("SELECT * FROM predictions ORDER BY id DESC")
    return cursor.fetchall()
