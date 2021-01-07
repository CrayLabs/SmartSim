import time

from ..constants import STATUS_NEW


class Job:
    """Keep track of various information for the controller.
    In doing so, continuously add various fields of information
    that is queryable by the user through interface methods in
    the controller class.
    """

    def __init__(self, job_name, job_id, entity):
        """Initialize a Job.

        :param job_name: The name of the job
        :type job_name: str
        :param job_id: The id associated with the job
        :type job_id: str
        :param entity: The SmartSim entity associated with the job
        :type entity: SmartSim Entity
        """
        self.name = job_name
        self.jid = job_id
        self.entity = entity
        self.status = STATUS_NEW
        self.raw_status = None
        self.returncode = None
        self.output = None
        self.error = None
        self.nodes = None
        self.start_time = time.time()
        self.history = History()

    def set_status(self, new_status, raw_status, returncode, error=None, output=None):
        """Set the status  of a job.

        :param new_status: The new status of the job
        :type new_status: str
        :param returncode: The return code for the job
        :type return_code: str
        """
        self.status = new_status
        self.raw_status = raw_status
        self.returncode = returncode
        self.error = error
        self.output = output

    def record_history(self):
        """Record the launching history of a job."""
        job_time = time.time() - self.start_time
        self.history.record(self.jid, self.status, self.returncode, job_time)

    def reset(self, new_job_id):
        """Reset the job in order to be able to restart it.

        :param new_job_id: new job id to launch under
        :type new_job_id: str
        """
        self.jid = new_job_id
        self.status = "NEW"
        self.returncode = None
        self.output = None
        self.error = None
        self.nodes = None
        self.start_time = time.time()
        self.history.new_run()

    def error_report(self):
        """A descriptive error report based on job fields

        :return: error report for display in terminal
        :rtype: str
        """
        warning = f"{self.name} failed. See below for details \n"
        if self.error:
            warning += f"{self.entity.type} {self.name} produced the following error \n"
            warning += f"Error: {self.error} \n"
        if self.output:
            warning += f"Output: {self.output} \n"
        warning += f"Job status at failure: {self.status} \n"
        warning += f"Launcher status at failure: {self.raw_status} \n"
        warning += f"Job returncode: {self.returncode} \n"
        warning += "For more information on the error, check the files below: \n"
        warning += f"{self.entity.type} error file: {self.entity.run_settings.get('err_file')} \n"
        warning += f"{self.entity.type} output file: {self.entity.run_settings.get('out_file')} \n"
        return warning

    def __str__(self):
        """Return user-readable string of the Job

        :returns: A user-readable string of the Job
        :rtype: str
        """
        job = "{}({}): {}"
        return job.format(self.name, self.jid, self.status)


class History:
    """History of a job instance. Holds various attributes based
    on the previous launches of a job.
    """

    def __init__(self, runs=0):
        """Init a history object for a job

        :param runs: number of runs so far, defaults to 0
        :type runs: int, optional
        """
        self.runs = runs
        self.jids = dict()
        self.statuses = dict()
        self.returns = dict()
        self.job_times = dict()

    def record(self, job_id, status, returncode, job_time):
        """record the history of a job"""
        self.jids[self.runs] = job_id
        self.statuses[self.runs] = status
        self.returns[self.runs] = returncode
        self.job_times[self.runs] = job_time

    def new_run(self):
        """increment run total"""
        self.runs += 1
