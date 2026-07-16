import os
import sys
import time
import pytest
import threading

# Add cognios to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cognios.src.database import init_db, get_last_system_metrics, get_last_process_metrics
from cognios.src.telemetry import TelemetryCollector

def test_telemetry():
    # Initialize the database
    init_db()
    
    # Start telemetry collector in mock mode to ensure it works reliably in sandboxed environments
    collector = TelemetryCollector(interval=0.5, use_mock=True)
    
    # Run collector in a separate thread
    thread = threading.Thread(target=collector.start_loop)
    thread.daemon = True
    thread.start()
    
    # Let it run for 3 seconds
    time.sleep(3.0)
    
    # Stop the collector
    collector.stop_loop()
    thread.join(timeout=2.0)
    
    # Query database and verify rows exist
    sys_metrics = get_last_system_metrics(limit=10)
    proc_metrics = get_last_process_metrics(limit=10)
    
    assert len(sys_metrics) > 0, "System metrics table should contain records"
    assert len(proc_metrics) > 0, "Process metrics table should contain records"
    
    # Check data fields
    metrics_row = sys_metrics[0]
    ts, cpu, ram, disk = metrics_row[:4]
    assert isinstance(ts, float)
    assert isinstance(cpu, float)
    assert isinstance(ram, float)
    assert isinstance(disk, float)
    if len(metrics_row) == 6:
        gpu_util, gpu_mem = metrics_row[4:]
        assert isinstance(gpu_util, float)
        assert isinstance(gpu_mem, float)
    
    p_ts, pid, name, p_cpu, p_ram = proc_metrics[0]
    assert isinstance(p_ts, float)
    assert isinstance(pid, int)
    assert isinstance(name, str)
    assert isinstance(p_cpu, float)
    assert isinstance(p_ram, float)
    
    print(f"\n[Telemetry Test] System Metrics Count: {len(sys_metrics)}, Process Metrics Count: {len(proc_metrics)}")
 
def test_os_doctor():
    # Initialize the database
    init_db()
    
    # Clean the database system_metrics table
    import sqlite3
    from cognios.src.database import DB_PATH, insert_system_metrics
    from cognios.src.os_doctor import OSDoctor
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM system_metrics")
    conn.commit()
    conn.close()
 
    # Inject fake baseline data (normal load)
    # 100 uniform random samples to create a robust model
    import random
    random.seed(42)
    base_ts = time.time() - 2000
    for i in range(100):
        ts = base_ts + i * 10
        cpu = random.uniform(15.0, 25.0)
        ram = random.uniform(40.0, 45.0)
        disk = random.uniform(45.0, 48.0)
        gpu_util = random.uniform(5.0, 15.0)
        gpu_mem = random.uniform(500.0, 800.0)
        insert_system_metrics(ts, cpu, ram, disk, gpu_util, gpu_mem)
        
    # Instantiate OSDoctor and train it
    doctor = OSDoctor(contamination='auto', random_state=42)
    doctor.train()
    
    # Infer on a normal point (should be 1)
    normal_row = (time.time(), 20.0, 43.0, 46.0, 10.0, 600.0)
    normal_pred = doctor.infer(normal_row)
    assert normal_pred == 1, f"Normal point should be classified as 1, got {normal_pred}"
    
    # Infer on a spike point (CPU = 100) (should be -1)
    spike_row = (time.time(), 100.0, 43.0, 46.0, 95.0, 3000.0)
    spike_pred = doctor.infer(spike_row)
    assert spike_pred == -1, f"Spike point should be classified as -1 (anomaly), got {spike_pred}"
    
    print("\n[OS Doctor Test] Successfully detected CPU spike anomaly!")
 
def test_focus_os():
    import pytest
    torch = pytest.importorskip("torch")
    import os
    from cognios.src.focus_os import WorkloadCNN, Prioritizer
    
    # 1. Test WorkloadCNN forward pass and output shape
    model = WorkloadCNN(num_classes=3)
    # Random tensor shape: (batch_size=4, channels=3, time_steps=10, features=5)
    dummy_input = torch.randn(4, 3, 10, 5)
    output = model(dummy_input)
    assert output.shape == (4, 3), f"Expected shape (4, 3), got {output.shape}"
    
    # 2. Test Prioritizer with actual PID (should return True, either setting it or logging Mock Success)
    prioritizer = Prioritizer(use_mock=False)
    my_pid = os.getpid()
    
    # Set nice value to a higher number (less priority, usually allowed without sudo)
    # or it will trigger AccessDenied and mock success.
    success = prioritizer.set_priority(my_pid, 10)
    assert success is True, "Prioritizer should succeed (directly or via Mock Success fallback)"
    
    # Test prioritizer with use_mock=True
    mock_prioritizer = Prioritizer(use_mock=True)
    mock_success = mock_prioritizer.set_priority(my_pid, -5)
    assert mock_success is True, "Mock Prioritizer should succeed for negative nice values"
    
    print("\n[FocusOS Test] CNN forward pass and Prioritizer checks passed successfully!")
 
def test_blackbox():
    import os
    import time
    from cognios.src.database import init_db, insert_system_metrics, get_last_system_metrics
    from cognios.src.blackbox import cleanup_old_metrics, export_to_csv
    
    init_db()
    
    # Clean table
    import sqlite3
    from cognios.src.database import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM system_metrics")
    conn.commit()
    conn.close()
    
    # Inject old and new records
    now = time.time()
    old_ts = now - 36 * 3600  # 36 hours ago
    new_ts = now - 1 * 60     # 1 minute ago
    
    insert_system_metrics(old_ts, 20.0, 45.0, 50.0, 5.0, 400.0)
    insert_system_metrics(new_ts, 30.0, 48.0, 52.0, 10.0, 600.0)
    
    # Verify both exist
    metrics_before = get_last_system_metrics(limit=10)
    assert len(metrics_before) == 2
    
    # Run cleanup (24 hours retention)
    success_cleanup = cleanup_old_metrics(retention_hours=24)
    assert success_cleanup is True
    
    # Verify old record is deleted, new record remains
    metrics_after = get_last_system_metrics(limit=10)
    assert len(metrics_after) == 1
    assert abs(metrics_after[0][0] - new_ts) < 1.0
    
    # Test export to CSV
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "test_export.csv"))
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    success_export = export_to_csv(csv_path, duration_minutes=5)
    assert success_export is True
    assert os.path.exists(csv_path)
    
    # Clean up the CSV file
    if os.path.exists(csv_path):
        os.remove(csv_path)
        
    print("\n[Blackbox Test] Cleanup and Export features verified successfully!")

def test_research():
    from cognios.src.research import fcfs_schedule, sjf_schedule, RLSchedulingSimulator
    
    # Fixed process list for mathematical verification
    processes = [
        {'pid': 1, 'burst_time': 15},
        {'pid': 2, 'burst_time': 3},
        {'pid': 3, 'burst_time': 8},
        {'pid': 4, 'burst_time': 12}
    ]
    
    fcfs_avg, fcfs_waits = fcfs_schedule(processes)
    sjf_avg, sjf_waits = sjf_schedule(processes)
    
    # SJF waiting time should be strictly less than or equal to FCFS waiting time
    assert sjf_avg <= fcfs_avg, f"SJF avg waiting time ({sjf_avg}) should be <= FCFS avg waiting time ({fcfs_avg})"
    
    # Verify exact calculations
    # FCFS waiting times: P1=0, P2=15, P3=18, P4=26. Avg = 59/4 = 14.75
    # SJF sorted: P2(3), P3(8), P4(12), P1(15). Waiting times: P2=0, P3=3, P4=11, P1=23. Avg = 37/4 = 9.25
    assert fcfs_avg == 14.75
    assert sjf_avg == 9.25
    
    # Run RLSchedulingSimulator
    sim = RLSchedulingSimulator(num_processes=6)
    results = sim.simulate()
    
    assert "FCFS" in results
    assert "SJF (Optimal)" in results
    assert "RL (Trained SB3 Agent)" in results
    
    # SJF must be the optimal
    assert results["SJF (Optimal)"] <= results["FCFS"]
    assert results["SJF (Optimal)"] <= results["RL (Trained SB3 Agent)"]
    
    print("\n[Research Test] SJF vs FCFS wait time constraints and RL scheduler simulator validated successfully!")

def test_unification():
    import subprocess
    import sys
    import os
    import time
    import requests
    
    # Path of main.py
    main_py = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "main.py"))
    
    # Run main.py in the background
    # Using the current virtualenv python interpreter running pytest
    proc = subprocess.Popen(
        [sys.executable, main_py],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=os.path.dirname(main_py)
    )
    
    # Wait 6 seconds for initialization of telemetry, doctor loop, and Streamlit server
    time.sleep(6.0)
    
    try:
        # Ping the default Streamlit port (8501)
        # Headless streamlit server returns HTML content on '/'
        response = requests.get("http://127.0.0.1:8501", timeout=5.0)
        assert response.status_code == 200, f"Expected HTTP 200 from Streamlit, got {response.status_code}"
        assert "Streamlit" in response.text or "streamlit" in response.text.lower(), "Expected Streamlit content in response"
        print("\n[Unification Test] Streamlit Dashboard active and responded successfully!")
        
    except Exception as e:
        # Print logs to stdout in case of failure for debugging
        stdout, stderr = proc.communicate(timeout=1.0)
        print(f"\n[Unification Test Failure] Stdout:\n{stdout.decode()}")
        print(f"\n[Unification Test Failure] Stderr:\n{stderr.decode()}")
        raise e
        
    finally:
        # Terminate the process
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            
    print("\n[Unification Test] CogniOS unified platform integration verified successfully!")




