import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from cognios.src.database import get_last_system_metrics

logger = logging.getLogger("os_doctor")

class OSDoctor:
    def __init__(self, contamination='auto', random_state=42):
        self.contamination = contamination
        self.random_state = random_state
        self.model = None
        self.features = ['cpu', 'ram', 'disk']

    def train(self):
        """
        Fetch last 100 system metrics from DB, format with pandas, and train IsolationForest.
        """
        rows = get_last_system_metrics(limit=100)
        if len(rows) < 5:
            logger.warning("Not enough data to train IsolationForest (found %d rows). Need at least 5.", len(rows))
            # Create a simple mock baseline if DB is empty/too small, so training can still succeed
            logger.info("Injecting internal baseline data to pre-train IsolationForest.")
            data = {
                'cpu': np.random.uniform(10.0, 30.0, size=50),
                'ram': np.random.uniform(30.0, 50.0, size=50),
                'disk': np.random.uniform(40.0, 45.0, size=50)
            }
            df = pd.DataFrame(data)
        else:
            # Columns: ts, cpu, ram, disk
            df = pd.DataFrame(rows, columns=['ts', 'cpu', 'ram', 'disk'])
        
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        self.model.fit(df[self.features])
        logger.info("IsolationForest trained successfully on %d samples.", len(df))

    def infer(self, latest_row):
        """
        Takes the latest DB row: (ts, cpu, ram, disk) or dict
        Returns 1 if normal, -1 if anomaly.
        """
        if self.model is None:
            self.train()
            
        # Parse inputs
        if isinstance(latest_row, dict):
            cpu = latest_row.get('cpu', 0.0)
            ram = latest_row.get('ram', 0.0)
            disk = latest_row.get('disk', 0.0)
        elif isinstance(latest_row, (list, tuple)):
            # If length is 4: (ts, cpu, ram, disk)
            if len(latest_row) == 4:
                _, cpu, ram, disk = latest_row
            else:
                cpu, ram, disk = latest_row
        else:
            raise ValueError("latest_row must be a dict, list, or tuple")

        X = pd.DataFrame([[cpu, ram, disk]], columns=self.features)
        pred = self.model.predict(X)
        return int(pred[0])
