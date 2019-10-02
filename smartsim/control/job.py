

class Job:
    """Keep track of various information for the controller.
       In doing so, continuously add various fields of information
       that is queriable by the user through interface methods in
       the controller class.
    """

    def __init__(self, job_name, job_id, obj):
        self.name = job_name
        self.jid = job_id
        self.obj = obj            # the model
        self.status = "NEW"
        self.return_code = None

    def get_job_id(self):
        return self.jid

    def set_status(self, new_status):
        self.status = new_status

    def set_return_code(self, code):
        self.return_code = code

    def __str__(self):
        job = ("{}({}): {}")
        return job.format(self.name, self.jid, self.status)
