import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import text
from core.database import get_engine, init_db

def run_migration():
    print("Initializing DB config...")
    engine = get_engine()
    
    with engine.connect() as conn:
        print("Running ALTER TABLE on qa_logs...")
        conn.execute(text("ALTER TABLE qa_logs ADD COLUMN IF NOT EXISTS session_id VARCHAR(100);"))
        conn.execute(text("ALTER TABLE qa_logs ADD COLUMN IF NOT EXISTS user_id INTEGER;"))
        conn.commit()
    print("Migration complete!")

if __name__ == "__main__":
    run_migration()
