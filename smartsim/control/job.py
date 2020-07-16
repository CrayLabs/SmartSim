

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
        self.status = "NEW"
        self.nodes = None
        self.returncode = None

    def set_status(self, new_status, returncode):
        """Set the status of a job.

        :param new_status: The new status of the job
        :type new_status: str
        :param returncode: The return code for the job
        :type return_code: str
        """
        self.status = new_status
        self.returncode = returncode

    def __str__(self):
        """Return user-readable string of the Job

        :returns: A user-readable string of the Job
        :rtype: str
        """
        job = ("{}({}): {}")
        return job.format(self.name, self.jid, self.status)
