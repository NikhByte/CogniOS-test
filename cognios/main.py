import time
import subprocess
import threading
import sys
import os
import signal
import logging

# Ensure parent directory is in sys.path to resolve cognios package correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cognios.src.database import init_db, get_last_system_metrics
from cognios.src.telemetry import TelemetryCollector
from cognios.src.os_doctor import OSDoctor

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")

# Global list of active processes/daemons to terminate on exit
daemons = []

def run_telemetry():
    """
    Run the telemetry collection loop in a daemon thread.
    """
    init_db()
    # We use use_mock=True by default to ensure perfect stability in sandboxed environments
    collector = TelemetryCollector(interval=1.0, use_mock=True)
    collector.start_loop()

def run_doctor():
    """
    Run the OS Doctor loop which periodically trains and checks for anomalies.
    """
    time.sleep(2)  # Give telemetry a moment to write initial data
    doctor = OSDoctor(contamination='auto', random_state=42)
    
    while True:
        try:
            doctor.train()
            rows = get_last_system_metrics(limit=1)
            if rows:
                latest = rows[0]
                pred = doctor.infer(latest)
                status = "🚨 ANOMALOUS WORKLOAD DETECTED" if pred == -1 else "🟢 SYSTEM NORMAL"
                logger.info("[OS Doctor Monitor] Current status: %s (CPU: %s%%, RAM: %s%%)", status, latest[1], latest[2])
        except Exception as e:
            logger.error("OS Doctor loop encounter error: %s", e)
        time.sleep(10)

def main():
    logger.info("Initializing CogniOS Platform...")
    init_db()
    
    # 1. Start Telemetry collector thread
    tel_thread = threading.Thread(target=run_telemetry, name="TelemetryCollector")
    tel_thread.daemon = True
    tel_thread.start()
    logger.info("Telemetry Collector loop started in background thread.")
    
    # 2. Start OS Doctor monitor thread
    doc_thread = threading.Thread(target=run_doctor, name="OSDoctorMonitor")
    doc_thread.daemon = True
    doc_thread.start()
    logger.info("OS Doctor loop started in background thread.")
    
    # 3. Start Streamlit UI Subprocess
    ui_script = os.path.abspath(os.path.join(os.path.dirname(__file__), "src", "ui.py"))
    
    # Locate the correct python/streamlit inside the virtual environment
    venv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv"))
    streamlit_path = os.path.join(venv_dir, "bin", "streamlit")
    
    if not os.path.exists(streamlit_path):
        streamlit_path = "streamlit"  # Fallback to path
        
    cmd = [
        streamlit_path,
        "run",
        ui_script,
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    logger.info("Starting Streamlit Dashboard Subprocess...")
    try:
        ui_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        daemons.append(ui_proc)
        logger.info("Streamlit UI successfully launched on http://0.0.0.0:8501")
    except Exception as e:
        logger.error("Failed to launch Streamlit Dashboard: %s", e)
        sys.exit(1)
        
    # Main process wait loop
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Termination signal received. Shutting down all CogniOS subsystems...")
    finally:
        # Clean up
        if ui_proc:
            logger.info("Terminating Streamlit dashboard...")
            ui_proc.terminate()
            try:
                ui_proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                ui_proc.kill()
        logger.info("CogniOS platform successfully stopped.")

if __name__ == "__main__":
    main()
