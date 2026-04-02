# save as test_db.py and run it
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER')}:"
    f"{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:"
    f"{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

try:
    engine = create_engine(DATABASE_URL)
    conn = engine.connect()
    print("✅ Connected to PostgreSQL successfully!")
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {e}")