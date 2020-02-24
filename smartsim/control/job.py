

class Job:
    """Keep track of various information for the controller.
       In doing so, continuously add various fields of information
       that is queriable by the user through interface methods in
       the controller class.
    """

    def __init__(self, job_name, job_id, obj):
        self.name = job_name
        self.jid = job_id
        self.obj = obj
        self.status = "NEW"
        self.nodes = None

    def get_job_id(self):
        return self.jid

    def set_status(self, new_status):
        if new_status == 1:
            self.status = "RUNNING"
        elif new_status == -1:
            self.status = "COMPLETE"
        else:
            self.status = new_status

    def __str__(self):
        job = ("{}({}): {}")
        return job.format(self.name, self.jid, self.status)
