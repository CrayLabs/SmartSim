
from .job import Job

class JobManager:
    """Holds all of the jobs launched by the Controller. Jobs are held
       in three dictionaries corresponding to their entity type. The class
       is callable and will return a single dictionary of all jobs.
    """

    def __init__(self):
        self.jobs = {}
        self.db_jobs = {}
        self.node_jobs = {}

    def __getitem__(self, job_name):
        if job_name in self.db_jobs.keys():
            return self.db_jobs[job_name]
        elif job_name in self.node_jobs.keys():
            return self.node_jobs[job_name]
        elif job_name in self.jobs.keys():
            return self.jobs[job_name]
        else:
            raise KeyError

    def __call__(self):
        all_jobs = {
            **self.jobs,
            **self.node_jobs,
            **self.db_jobs
            }
        return all_jobs

    def add_job(self, name, job_id, entity):
        job = Job(name, job_id, entity)
        if entity.type == "db":
            self.db_jobs[name] = job
        elif entity.type == "node":
            self.node_jobs[name] = job
        else:
            self.jobs[name] = job

    def get_db_nodes(self):
        """Return a list of database nodes for cluster creation"""
        nodes = []
        for db_job in self.db_jobs.values():
            nodes.extend(db_job.nodes)
        return nodes
