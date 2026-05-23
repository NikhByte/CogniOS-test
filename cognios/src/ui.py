import streamlit as st
import pandas as pd
import time
import os
import sys

# Ensure cognios is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from cognios.src.database import get_last_system_metrics, get_last_process_metrics, init_db
from cognios.src.os_doctor import OSDoctor
from cognios.src.blackbox import cleanup_old_metrics, export_to_csv
from cognios.src.research import RLSchedulingSimulator

# Set Page Config
st.set_page_config(
    page_title="CogniOS | System Monitor & Workload Optimizer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS for styling and dark mode
st.markdown("""
<style>
    /* Premium fonts and background styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Outfit:wght@400;600;800&display=swap');
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background-color: #0d0f14;
        color: #e2e8f0;
    }
    
    /* Header Card styling */
    .header-card {
        background: linear-gradient(135deg, #1e1b4b 0%, #311042 100%);
        border: 1px solid #312e81;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    .header-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        margin: 0;
        background: linear-gradient(to right, #6366f1, #d946ef, #38bdf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #94a3b8;
        margin-top: 0.5rem;
    }
    
    /* Metrics panel card styling */
    .metric-card {
        background: rgba(22, 28, 41, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    /* Anomaly Banner styling */
    .anomaly-banner-healthy {
        background: linear-gradient(90deg, rgba(6, 78, 59, 0.8) 0%, rgba(4, 120, 87, 0.8) 100%);
        border: 1px solid #059669;
        border-radius: 8px;
        padding: 1rem;
        color: #a7f3d0;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
        animation: pulse 2s infinite;
    }
    
    .anomaly-banner-danger {
        background: linear-gradient(90deg, rgba(153, 27, 27, 0.8) 0%, rgba(185, 28, 28, 0.8) 100%);
        border: 1px solid #dc2626;
        border-radius: 8px;
        padding: 1rem;
        color: #fca5a5;
        font-weight: 600;
        text-align: center;
        margin-bottom: 1.5rem;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 0.9; }
        50% { opacity: 1; }
        100% { opacity: 0.9; }
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown("""
<div class="header-card">
    <h1 class="header-title">🧠 CogniOS</h1>
    <div class="header-subtitle">AI-Orchestrated System Telemetry & Workload Prioritization Engine</div>
</div>
""", unsafe_allow_html=True)

# Tabs
tab_telemetry, tab_doctor, tab_blackbox, tab_research = st.tabs([
    "📊 Live Telemetry",
    "🩺 OS Doctor (Anomaly Detection)",
    "📦 BlackBox & Cleanup",
    "🔬 Scheduler Research"
])

# Initialize DB if needed
init_db()

# Telemetry Tab
with tab_telemetry:
    st.subheader("System Performance telemetry")
    
    # Fetch last metrics
    sys_metrics = get_last_system_metrics(limit=60)
    proc_metrics = get_last_process_metrics(limit=15)
    
    if sys_metrics:
        df_sys = pd.DataFrame(sys_metrics, columns=['ts', 'cpu', 'ram', 'disk'])
        df_sys['time'] = pd.to_datetime(df_sys['ts'], unit='s').dt.strftime('%H:%M:%S')
        
        # Display main metric cards
        latest = df_sys.iloc[-1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">System CPU Usage</div>
                <div class="metric-value" style="color: #818cf8;">{latest['cpu']}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Memory Allocation</div>
                <div class="metric-value" style="color: #ec4899;">{latest['ram']}%</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Disk Storage Capacity</div>
                <div class="metric-value" style="color: #38bdf8;">{latest['disk']}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.write("")
        
        # Performance Charts
        st.subheader("Usage Trends")
        chart_df = df_sys.set_index('time')[['cpu', 'ram']]
        st.line_chart(chart_df, color=["#818cf8", "#ec4899"])
    else:
        st.info("No system metrics collected yet. Please ensure the telemetry collector daemon is running.")

    # Top processes
    st.subheader("Resource Intensive Processes")
    if proc_metrics:
        df_proc = pd.DataFrame(proc_metrics, columns=['ts', 'pid', 'name', 'cpu', 'ram'])
        # Sort and get unique latest PIDs
        df_proc_latest = df_proc.sort_values('ts').groupby('pid').last().reset_index()
        df_proc_latest = df_proc_latest.sort_values(by='cpu', ascending=False).head(10)
        
        st.dataframe(
            df_proc_latest[['pid', 'name', 'cpu', 'ram']],
            column_config={
                "pid": "Process ID",
                "name": "Process Name",
                "cpu": st.column_config.ProgressColumn("CPU Usage", format="%.1f%%", min_value=0, max_value=100),
                "ram": st.column_config.ProgressColumn("RAM Usage", format="%.1f%%", min_value=0, max_value=100)
            },
            hide_index=True,
            use_container_width=True
        )
    else:
        st.info("No process telemetry collected yet.")

# OS Doctor Tab
with tab_doctor:
    st.subheader("AI Anomaly Detection Diagnostics")
    
    sys_metrics = get_last_system_metrics(limit=100)
    
    if sys_metrics:
        # Train and infer
        doctor = OSDoctor(contamination='auto', random_state=42)
        with st.spinner("Analyzing baseline performance history..."):
            doctor.train()
            
        latest_row = sys_metrics[-1]
        prediction = doctor.infer(latest_row)
        
        if prediction == 1:
            st.markdown("""
            <div class="anomaly-banner-healthy">
                🟢 SYSTEM HEALTH STATUS: NORMAL (No Anomalies Detected)
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="anomaly-banner-danger">
                🚨 DETECTED ANOMALOUS SYSTEM BEHAVIOR (Workload Spike or Resource Exhaustion)
            </div>
            """, unsafe_allow_html=True)
            
        st.write("")
        st.markdown("### Isolation Forest Telemetry Profile")
        
        # Diagnostic columns
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Latest Sampled CPU", f"{latest_row[1]}%")
            st.metric("Latest Sampled RAM", f"{latest_row[2]}%")
            st.metric("Latest Sampled Disk", f"{latest_row[3]}%")
        with col2:
            st.markdown("""
            An **Isolation Forest** model isolates anomalies by randomly partitioning feature spaces.
            Anomalous workloads require fewer splits to isolate, resulting in shorter paths in the decision trees.
            - **Status**: Active
            - **Features Tracked**: `['cpu', 'ram', 'disk']`
            - **Diagnosis Interval**: Real-time
            """)
    else:
        st.info("Insufficient telemetry data available to perform diagnostics.")

# BlackBox Tab
with tab_blackbox:
    st.subheader("System Logs Archive & Database Actions")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Database Retention Management")
        st.write("Maintain SQLite performance by removing historical data logs older than 24 hours.")
        if st.button("🧼 Trigger Manual Database Cleanup"):
            success = cleanup_old_metrics(retention_hours=24)
            if success:
                st.success("Successfully purged metrics older than 24 hours from the database!")
            else:
                st.error("Purge execution failed. Check terminal logs.")
                
    with col2:
        st.markdown("#### Export Metrics")
        st.write("Generate a CSV snapshot of the last 5 minutes of telemetry metrics.")
        
        csv_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "metrics_dump.csv"))
        if st.button("📤 Export Last 5 mins to CSV"):
            success = export_to_csv(csv_filepath, duration_minutes=5)
            if success and os.path.exists(csv_filepath):
                with open(csv_filepath, "r") as f:
                    csv_data = f.read()
                st.download_button(
                    label="💾 Download CSV File",
                    data=csv_data,
                    file_name="cognios_telemetry_dump.csv",
                    mime="text/csv"
                )
                # Cleanup exported local copy
                os.remove(csv_filepath)
            else:
                st.error("Telemetry export execution failed.")

# Scheduler Research Tab
with tab_research:
    st.subheader("Scheduling Algorithms Simulator (Classic vs RL)")
    
    st.markdown("""
    This simulator models process scheduling waiting times across different policies.
    It compares **First Come First Serve (FCFS)**, **Shortest Job First (SJF)**, and a **Deep Reinforcement Learning (RL)** scheduler.
    """)
    
    # Run simulation
    if st.button("⚡ Run Scheduling Comparison Simulation"):
        sim = RLSchedulingSimulator(num_processes=8)
        results = sim.simulate()
        
        # Display results in columns
        cols = st.columns(4)
        colors = ["#e2e8f0", "#10b981", "#ef4444", "#3b82f6"]
        for idx, (policy, val) in enumerate(results.items()):
            with cols[idx]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">{policy}</div>
                    <div class="metric-value" style="color: {colors[idx]};">{val}s</div>
                </div>
                """, unsafe_allow_html=True)
                
        # Bar chart
        st.write("")
        st.subheader("Average Waiting Time (Lower is Better)")
        df_results = pd.DataFrame(list(results.items()), columns=['Policy', 'Avg Wait Time (Seconds)'])
        st.bar_chart(df_results.set_index('Policy'))
        
        st.markdown("""
        > [!NOTE]
        > **Shortest Job First (SJF)** represents the mathematical lower bound for average wait time when processes arrive concurrently. 
        > The **Stable-Baselines3 RL agent** learns process burst structures and dynamically optimizes sequence sorting to match or exceed static heuristic bounds on varying workloads.
        """)
