import os
import sqlite3

def init_db():
    # Get the directory where the script is located
    script_directory = os.path.dirname(__file__)
    
    # Define the database path relative to the script directory
    db_path = os.path.join(script_directory, 'trades.db')
    
    # Create and initialize the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create the trades table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            current_price REAL,
            lstm_prediction REAL,
            rf_prediction REAL,
            ensemble_prediction REAL,
            action TEXT
        )
    ''')
    
    # Commit and close the connection
    conn.commit()
    conn.close()