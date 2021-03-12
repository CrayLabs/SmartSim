from collections import namedtuple

class StepMapping:

    StepMap = namedtuple("StepMap", ["step_id", "task_id", "managed"])

    def __init__(self):
        # step_name : wlm_id, pid, wlm_managed?
        self.mapping = {}

    def __getitem__(self, step_name):
        return self.mapping[step_name]

    def add(self, step_name, step_id=None, task_id=None, managed=True):
        self.mapping[step_name] = self.StepMap(step_id, task_id, managed)

    def get_task_id(self, step_id):
        """Get the task id from the step id"""
        task_id = None
        for stepmap in self.mapping.values():
            if stepmap.step_id == step_id:
                task_id = stepmap.task_id
                break
        return task_id

    def get_ids(self, step_names, managed=True):
        ids = []
        for name in step_names:
            if name in self.mapping:
                stepmap = self.mapping[name]
                # do we want task(unmanaged) or step(managed) id?
                if managed and stepmap.managed:
                    ids.append(stepmap.step_id)
                elif not managed and not stepmap.managed:
                    ids.append(stepmap.task_id)
        return ids
