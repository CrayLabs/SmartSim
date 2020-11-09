import coloredlogs

# Constants for SmartSim
# Do not change these unless you know what you are doing. :)


# Intervals for Job Manager that depend on launcher
# For WLM, we don't want to ping the wlm too much
LOCAL_JM_INTERVAL = 2
WLM_JM_INTERVAL = 5

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

# constants for logging
coloredlogs.DEFAULT_DATE_FORMAT = "%H:%M:%S"
coloredlogs.DEFAULT_LOG_FORMAT = (
    "%(asctime)s %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s"
)
# optional thread name logging for debugging
# coloredlogs.DEFAULT_LOG_FORMAT = '%(asctime)s [%(threadName)s] %(hostname)s %(name)s[%(process)d] %(levelname)s %(message)s'
