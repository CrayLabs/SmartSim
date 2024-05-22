from smartsim.jobgrouphold.baseJobGroup import BaseJobGroup
from smartsim.jobgrouphold.jobGroup import JobGroup, BaseJob

def test_create_JobGroup():
    job_1 = BaseJob()
    job_group = JobGroup([job_1])
    assert len(job_group) == 1

def test_getitem_JobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = JobGroup([job_1,job_2])
    get_value = job_group[0]
    assert get_value == job_1

# cannot test setitem until Job is implemented since there is no comparison bc of the deep copy
# def test_setitem_JobGroup():
#     job_1 = BaseJob()
#     job_2 = BaseJob()
#     job_group = JobGroup([job_1,job_2])
#     job_3 = BaseJob()
#     job_group[1] = job_3
#     get_value = job_group[1]
#     assert get_value == job_3

def test_delitem_JobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = JobGroup([job_1,job_2])
    assert len(job_group) == 2
    del(job_group[1])
    assert len(job_group) == 1

def test_len_JobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = JobGroup([job_1,job_2])
    assert len(job_group) == 2

def test_insert_JobGroup():
    job_1 = BaseJob()
    job_2 = BaseJob()
    job_group = JobGroup([job_1,job_2])
    job_3 = BaseJob()
    job_group.insert(0, job_3)
    get_value = job_group[0]
    assert get_value == job_3