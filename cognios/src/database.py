import sqlite3
import os
import logging

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "cognios.db"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("database")

def get_db_connection():
    """
    Connect to the SQLite database with robust concurrency settings:
    - check_same_thread=False (allow multi-threaded access)
    - timeout=30.0 (wait up to 30s for lock release)
    - journal_mode=WAL (Write-Ahead Logging for concurrent readers/writers)
    - synchronous=NORMAL
    """
    conn = sqlite3.connect(DB_PATH, timeout=30.0, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    """
    Initialize the SQLite database with system_metrics and process_metrics tables.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # System metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS system_metrics (
            ts REAL,
            cpu REAL,
            ram REAL,
            disk REAL
        )
    """)
    
    # Process metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS process_metrics (
            ts REAL,
            pid INTEGER,
            name TEXT,
            cpu REAL,
            ram REAL
        )
    """)
    
    # Ensure columns exist in system_metrics for GPU telemetry
    cursor.execute("PRAGMA table_info(system_metrics)")
    columns = [col[1] for col in cursor.fetchall()]
    if 'gpu_util' not in columns:
        cursor.execute("ALTER TABLE system_metrics ADD COLUMN gpu_util REAL DEFAULT 0.0")
        logger.info("Added gpu_util column to system_metrics database table.")
    if 'gpu_mem' not in columns:
        cursor.execute("ALTER TABLE system_metrics ADD COLUMN gpu_mem REAL DEFAULT 0.0")
        logger.info("Added gpu_mem column to system_metrics database table.")
        
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully at %s", DB_PATH)

def insert_system_metrics(ts, cpu, ram, disk, gpu_util=0.0, gpu_mem=0.0):
    """
    Insert a single row of system metrics including GPU metrics.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO system_metrics (ts, cpu, ram, disk, gpu_util, gpu_mem) VALUES (?, ?, ?, ?, ?, ?)",
            (ts, cpu, ram, disk, gpu_util, gpu_mem)
        )
        conn.commit()
    except Exception as e:
        logger.error("Failed to insert system metrics: %s", e)
    finally:
        conn.close()

def insert_process_metrics(metrics_list):
    """
    Batch-insert multiple process metrics.
    metrics_list: list of tuples (ts, pid, name, cpu, ram)
    """
    if not metrics_list:
        return
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.executemany(
            "INSERT INTO process_metrics (ts, pid, name, cpu, ram) VALUES (?, ?, ?, ?, ?)",
            metrics_list
        )
        conn.commit()
    except Exception as e:
        logger.error("Failed to batch insert process metrics: %s", e)
    finally:
        conn.close()

def get_last_system_metrics(limit=100):
    """
    Fetch the last `limit` rows of system metrics sorted by timestamp descending.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ts, cpu, ram, disk, gpu_util, gpu_mem FROM system_metrics ORDER BY ts DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        # Return sorted by timestamp ascending
        return sorted(rows, key=lambda x: x[0])
    except Exception as e:
        logger.error("Failed to fetch system metrics: %s", e)
        return []
    finally:
        conn.close()

def get_last_process_metrics(limit=100):
    """
    Fetch the last `limit` rows of process metrics sorted by timestamp descending.
    """
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ts, pid, name, cpu, ram FROM process_metrics ORDER BY ts DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        return sorted(rows, key=lambda x: x[0])
    except Exception as e:
        logger.error("Failed to fetch process metrics: %s", e)
        return []
    finally:
        conn.close()
