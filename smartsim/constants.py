# Constants for SmartSim

# Interval for Job Manager
LOCAL_JM_INTERVAL = 2

# Task Manager Interval
TM_INTERVAL = 1

# Statuses that are applied to jobs
STATUS_RUNNING = "Running"
STATUS_COMPLETED = "Completed"
STATUS_CANCELLED = "Cancelled"
STATUS_FAILED = "Failed"
STATUS_NEW = "New"
STATUS_PAUSED = "Paused"

# SmartSim status mapping
SMARTSIM_STATUS = {
    "Paused": STATUS_PAUSED,
    "Completed": STATUS_COMPLETED,
    "Cancelled": STATUS_CANCELLED,
    "Failed": STATUS_FAILED,
    "New": STATUS_NEW,
}

# Status groupings
TERMINAL_STATUSES = (STATUS_CANCELLED, STATUS_COMPLETED, STATUS_FAILED)
LIVE_STATUSES = (STATUS_RUNNING, STATUS_PAUSED, STATUS_NEW)
