# CogniOS

CogniOS is a lightweight, self-contained system telemetry monitor and workload optimization platform written in Python. It provides high-concurrency performance metrics logging, Isolation Forest anomaly detection, neural network-based workload classification, and process priority adjustments with safety fallbacks for sandboxed environments.

The project is built using Test-Driven Development (TDD) and features a dynamic dark-themed dashboard.

---

## 📂 Project Structure

```
/cognios
  ├── /src
  │    ├── __init__.py
  │    ├── database.py       # Thread-safe SQLite connection manager (WAL mode)
  │    ├── telemetry.py      # Real-time telemetry loop & stateful OS simulation
  │    ├── os_doctor.py      # Isolation Forest anomaly detection classifier
  │    ├── focus_os.py       # PyTorch CNN workload classifier & process prioritizer
  │    ├── blackbox.py       # Historical logs database cleanup & CSV exporter
  │    ├── research.py       # FCFS vs SJF waiting time and RL scheduler simulator
  │    └── ui.py             # Streamlit graphical dashboard
  ├── /tests
  │    └── test_all.py       # Pytest suite verifying all core modules
  ├── requirements.txt       # Project dependencies
  └── main.py                # Platform orchestrator (starts daemons and UI)
```

---

## ⚡ Core Features

*   **Non-Blocking Telemetry DB**: Uses SQLite configured with Write-Ahead Logging (`WAL` mode) and a `NORMAL` synchronous profile. This setup supports concurrent, thread-safe writes from the telemetry collector while Streamlit and the ML diagnostic modules continuously read the data.
*   **Stateful System Simulator**: When real OS metrics are restricted, the telemetry engine runs a state machine mimicking transitions between `'idle'` operations and heavy `'spike'` workloads (e.g. active code compilation) with realistic random-walk momentum and slow-drifting disk allocation.
*   **ML Diagnostics (OS Doctor)**: An unsupervised `IsolationForest` continuously trains on active telemetry to distinguish between healthy baselines and high-intensity resource spikes.
*   **Workload Classification & Nice Control (FocusOS)**:
    *   Features a PyTorch Convolutional Neural Network (`WorkloadCNN`) that maps multi-channel temporal metric windows `(channels=3, time_steps=10, features=5)` to active execution states.
    *   Contains a `nice` value scheduler with automated `psutil.AccessDenied` exception handling to safely mock optimizations in sandboxed or non-sudo setups.
*   **Scheduling Simulator**: Evaluates queue waiting times across First Come First Serve (FCFS), mathematically optimal Shortest Job First (SJF), and simulated Reinforcement Learning (RL) scheduling policies.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+ (Tested on Python 3.14)
*   Virtual environment tools (`venv`)

### Installation & Setup

1.  Clone the repository and navigate to the project directory:
    ```bash
    cd CogniOS
    ```

2.  Initialize and activate a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3.  Install dependencies:
    ```bash
    pip install -r cognios/requirements.txt
    ```

### Running the Platform

To start the unified platform (starts the telemetry daemon, anomaly monitor, and the graphical Streamlit dashboard):

```bash
python cognios/main.py
```

Once running, access the interactive dashboard at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

## 🧪 Testing

The platform includes a comprehensive pytest suite covering telemetry capture, ML diagnostics, neural net tensor passes, SQLite concurrent writes, database pruning, and UI integration.

To run the complete test suite:

```bash
pytest cognios/tests/test_all.py -v
```

---

## 🛠️ Subsystem Architecture

```mermaid
graph TD
    A[main.py: Platform Orchestrator] --> B[telemetry.py: Telemetry Daemon]
    A --> C[os_doctor.py: ML Anomaly Detector]
    A --> D[ui.py: Streamlit Dashboard Subprocess]
    
    B -->|Batch Insert (psutil/mock)| E[(cognios.db: SQLite WAL)]
    C -->|Train / Read| E
    D -->|Plot / Read| E
    
    subgraph "Workload Optimization"
        F[focus_os.py: WorkloadCNN]
        G[focus_os.py: Process Prioritizer]
    end
    
    subgraph "Logs & Cleanup"
        H[blackbox.py: DB Purge & CSV Exporter]
    end
    
    subgraph "Scheduler Research"
        I[research.py: FCFS vs SJF vs RL Simulator]
    end
```
