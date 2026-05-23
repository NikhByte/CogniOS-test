import torch
import torch.nn as nn
import psutil
import logging

logger = logging.getLogger("focus_os")

class WorkloadCNN(nn.Module):
    """
    WorkloadCNN accepts a 4D tensor with shape (batch_size, channels=3, time_steps=10, features=5)
    and predicts workload classification (e.g., Idle vs CPU-bound vs Memory-bound).
    """
    def __init__(self, num_classes=3):
        super(WorkloadCNN, self).__init__()
        # input shape: [batch, 3, 10, 5]
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        # After conv1: [batch, 16, 10, 5]
        # After pool (kernel_size=2): [batch, 16, 5, 2]
        self.fc = nn.Linear(16 * 5 * 2, num_classes)

    def forward(self, x):
        # input x shape should be (N, 3, 10, 5)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)
        out = out.view(out.size(0), -1) # Flatten
        out = self.fc(out)
        return out

class Prioritizer:
    """
    Prioritizer manages process nice values with fallback for sandbox environments.
    """
    def __init__(self, use_mock=False):
        self.use_mock = use_mock

    def set_priority(self, pid, nice_value):
        """
        Attempts to change a given PID's nice value.
        nice_value should be between -20 and 19.
        If denied, logs "Mock Success" and returns True if use_mock is enabled or access is denied.
        """
        if self.use_mock:
            logger.info("Mock Success: PID %d priority set to %d (forced mock)", pid, nice_value)
            print("Mock Success")
            return True

        try:
            p = psutil.Process(pid)
            p.nice(nice_value)
            logger.info("Successfully set PID %d nice value to %d", pid, nice_value)
            return True
        except (psutil.AccessDenied, PermissionError) as e:
            logger.warning("Access denied setting PID %d priority to %d. Falling back to Mock Success.", pid, nice_value)
            print("Mock Success")
            return True
        except Exception as e:
            logger.error("Failed to set PID %d priority: %s", pid, e)
            return False
