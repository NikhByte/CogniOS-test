import time
import psutil
import logging
import random
from cognios.src.database import insert_system_metrics, insert_process_metrics

logger = logging.getLogger("telemetry")

class TelemetryCollector:
    def __init__(self, interval=1.0, use_mock=False):
        self.interval = interval
        self.use_mock = use_mock
        self._running = False
        
        # Stateful realistic mock data metrics
        self.mock_cpu = 15.0
        self.mock_ram = 42.0
        self.mock_disk = 45.34
        self.mock_state = 'idle'  # 'idle' or 'spike'
        self.mock_state_ticks = 0

    def _update_mock_state(self):
        """
        Updates the internal workload simulation state.
        Simulates an OS transitioning between idle states and heavy compilation/workload spikes.
        """
        self.mock_state_ticks += 1
        
        if self.mock_state == 'idle':
            # Idle states last at least 15 ticks, then have a 8% chance to trigger a workload spike
            if self.mock_state_ticks > 15 and random.random() < 0.08:
                self.mock_state = 'spike'
                self.mock_state_ticks = 0
        elif self.mock_state == 'spike':
            # Spikes represent active processes and last around 8-12 ticks, then drop back to idle
            if self.mock_state_ticks > random.randint(8, 12):
                self.mock_state = 'idle'
                self.mock_state_ticks = 0

    def collect_system_metrics(self):
        """
        Collect system-wide CPU, RAM, and Disk metrics.
        Returns a tuple: (ts, cpu, ram, disk)
        """
        ts = time.time()
        
        # Always run mock state machine if use_mock is active or as a fallback
        self._update_mock_state()
        
        if self.use_mock:
            # Simulate CPU with momentum based on active workload state
            if self.mock_state == 'idle':
                target = random.uniform(8.0, 18.0)
                step = random.uniform(-1.5, 1.5)
                self.mock_cpu += (target - self.mock_cpu) * 0.15 + step
            else: # 'spike'
                target = random.uniform(78.0, 92.0)
                step = random.uniform(-2.5, 2.5)
                self.mock_cpu += (target - self.mock_cpu) * 0.35 + step
                
            self.mock_cpu = max(5.0, min(98.0, self.mock_cpu))
            
            # Simulate RAM slowly shifting to reflect the workload pressure
            if self.mock_state == 'spike':
                self.mock_ram += random.uniform(0.1, 0.4)
            else:
                self.mock_ram -= random.uniform(0.05, 0.2)
            self.mock_ram = max(30.0, min(85.0, self.mock_ram))
            
            # Simulate Disk usage with extremely slow drift (realistic for disk capacity!)
            # Disk space doesn't jump randomly by 10% every second.
            self.mock_disk += random.uniform(-0.002, 0.005)
            self.mock_disk = max(10.0, min(95.0, self.mock_disk))
            
            return ts, round(self.mock_cpu, 2), round(self.mock_ram, 2), round(self.mock_disk, 2)

        try:
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
            disk = psutil.disk_usage('/').percent
            return ts, cpu, ram, disk
        except Exception as e:
            logger.warning("Error reading real system metrics, falling back to stateful mock: %s", e)
            # Safe stateful fallback
            if self.mock_state == 'idle':
                self.mock_cpu += (random.uniform(8.0, 18.0) - self.mock_cpu) * 0.15 + random.uniform(-1.0, 1.0)
                self.mock_ram -= random.uniform(0.05, 0.2)
            else:
                self.mock_cpu += (random.uniform(78.0, 92.0) - self.mock_cpu) * 0.35 + random.uniform(-2.0, 2.0)
                self.mock_ram += random.uniform(0.1, 0.4)
            self.mock_cpu = max(5.0, min(98.0, self.mock_cpu))
            self.mock_ram = max(30.0, min(85.0, self.mock_ram))
            self.mock_disk += random.uniform(-0.002, 0.005)
            return ts, round(self.mock_cpu, 2), round(self.mock_ram, 2), round(self.mock_disk, 2)

    def collect_process_metrics(self):
        """
        Collect metrics for top processes by CPU and RAM usage.
        Returns a list of tuples: (ts, pid, name, cpu, ram)
        """
        ts = time.time()
        if self.use_mock:
            # Generate mock processes whose load corresponds directly to system load
            # This is mathematically consistent and extremely realistic!
            cpu_val = self.mock_cpu
            
            py3_cpu = round(cpu_val * random.uniform(0.55, 0.70) if self.mock_state == 'spike' else random.uniform(3.0, 8.0), 2)
            streamlit_cpu = round(cpu_val * random.uniform(0.08, 0.12) + 2.0, 2)
            sqlite3_cpu = round(cpu_val * random.uniform(0.05, 0.10) if self.mock_state == 'spike' else random.uniform(0.1, 1.5), 2)
            postgres_cpu = round(random.uniform(0.5, 2.5), 2)
            nginx_cpu = round(random.uniform(0.1, 1.0), 2)
            
            mock_processes = [
                (ts, 1001, "python3", py3_cpu, round(self.mock_ram * 0.12, 2)),
                (ts, 1002, "streamlit", streamlit_cpu, round(self.mock_ram * 0.18, 2)),
                (ts, 1003, "postgres", postgres_cpu, round(self.mock_ram * 0.08, 2)),
                (ts, 1004, "nginx", nginx_cpu, round(self.mock_ram * 0.04, 2)),
                (ts, 1005, "sqlite3", sqlite3_cpu, round(self.mock_ram * 0.03, 2))
            ]
            return mock_processes

        processes_data = []
        try:
            # Iterate through processes
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                try:
                    info = proc.info
                    pid = info['pid']
                    name = info['name'] or "Unknown"
                    cpu = info['cpu_percent'] or 0.0
                    ram = info['memory_percent'] or 0.0
                    processes_data.append((ts, pid, name, cpu, ram))
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue

            # Sort by CPU + RAM and return top 15 processes
            processes_data.sort(key=lambda x: x[3] + x[4], reverse=True)
            return processes_data[:15]
        except Exception as e:
            logger.warning("Error reading real process metrics, falling back to mock: %s", e)
            return [
                (ts, 1001, "python3_mock", round(random.uniform(5.0, 30.0), 2), round(random.uniform(2.0, 8.0), 2)),
                (ts, 1002, "streamlit_mock", round(random.uniform(1.0, 15.0), 2), round(random.uniform(3.0, 12.0), 2))
            ]

    def single_step(self):
        """
        Perform a single collection and insertion step.
        """
        ts, cpu, ram, disk = self.collect_system_metrics()
        insert_system_metrics(ts, cpu, ram, disk)
        
        proc_metrics = self.collect_process_metrics()
        insert_process_metrics(proc_metrics)

    def start_loop(self):
        """
        Start the telemetry collection loop.
        """
        self._running = True
        logger.info("Starting telemetry collection loop (interval=%ss, use_mock=%s)", self.interval, self.use_mock)
        # Call cpu_percent once to initialize
        if not self.use_mock:
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
                
        while self._running:
            start_time = time.time()
            try:
                self.single_step()
            except Exception as e:
                logger.error("Error in telemetry collection step: %s", e)
            
            elapsed = time.time() - start_time
            sleep_time = max(0.1, self.interval - elapsed)
            time.sleep(sleep_time)

    def stop_loop(self):
        """
        Stop the telemetry collection loop.
        """
        self._running = False
        logger.info("Telemetry collection loop stopped.")
