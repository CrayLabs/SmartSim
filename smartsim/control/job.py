

class Job:
    """Keep track of various information for the controller.
       In doing so, continuously add various fields of information
       that is queriable by the user through interface methods in
       the controller class.
    """

    def __init__(self, job_name, job_id, entity):
        self.name = job_name
        self.jid = job_id
        self.entity = entity
        self.status = "NEW"
        self.nodes = None
        self.returncode = None

    def get_job_id(self):
        return self.jid

    def set_status(self, new_status, returncode):
        self.status = new_status
        self.returncode = returncode

    def __str__(self):
        job = ("{}({}): {}")
        return job.format(self.name, self.jid, self.status)
