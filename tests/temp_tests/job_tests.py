from smartsim.jobgrouphold.baseJobGroup import BaseJobGroup
from smartsim.jobgrouphold.jobGroup import JobGroup, BaseJob

def test_jobs_property():
    job = BaseJob()
    job_group = JobGroup([job])
    assert len(job_group) == 1