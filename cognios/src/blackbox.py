import time
import os
import pandas as pd
import logging
from cognios.src.database import get_db_connection

logger = logging.getLogger("blackbox")

def cleanup_old_metrics(retention_hours=24):
    """
    Deletes metrics older than retention_hours (default 24h).
    """
    cutoff = time.time() - retention_hours * 3600
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM system_metrics WHERE ts < ?", (cutoff,))
        cursor.execute("DELETE FROM process_metrics WHERE ts < ?", (cutoff,))
        conn.commit()
        logger.info("Database cleaned up: metrics older than %d hours deleted.", retention_hours)
        return True
    except Exception as e:
        logger.error("Failed to clean up database: %s", e)
        return False
    finally:
        conn.close()

def export_to_csv(filepath, duration_minutes=5):
    """
    Export system metrics from the last `duration_minutes` to a CSV file.
    """
    cutoff = time.time() - duration_minutes * 60
    conn = get_db_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(system_metrics)")
        cols = [col[1] for col in cursor.fetchall()]
        col_str = ", ".join(cols)
        
        query = f"SELECT {col_str} FROM system_metrics WHERE ts >= ? ORDER BY ts ASC"
        df = pd.read_sql_query(query, conn, params=(cutoff,))
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info("Exported last %d minutes of system metrics to %s", duration_minutes, filepath)
        return True
    except Exception as e:
        logger.error("Failed to export metrics to CSV: %s", e)
        return False
    finally:
        conn.close()
