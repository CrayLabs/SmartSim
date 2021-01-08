import abc

class Launcher(abc.ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()

    # -------------- Abstract Methods --------------

    @abc.abstractmethod
    def create_step(self, entity_name, run_settings, multi_prog=False):
        """Convert a smartsim entity run_settings into a job step
           to be launched on an allocation. An entity must have an
           allocation assigned to it in the running settings or
           create_step will throw a LauncherError

        :param entity_name: name of the step to be launch, usually entity.name
        :type entity_name: str
        :param run_settings: smartsim run_settings for an entity
        :type run_settings: dict
        :param multi_prog: create a multi-program step, defaults to False
        :type multi_prog: bool, optional
        :raises LauncherError: if step creation fails
        :return: Step object
        """

    @abc.abstractmethod
    def get_step_status(self, step_id):
        """Return the status of a job step from either the OS or
           the workload manager.

        :param step_id: id of the step in the form of xxxxxx.x
        :type step_id: str
        :return: status of the job step and returncode
        :rtype: StepInfo
        """

    @abc.abstractmethod
    def get_step_update(self, step_ids):
        """Get status updates of all steps at once

        :param step_ids: list of step_ids (str)
        :type step_ids: list
        :return: list of StepInfo for update
        :rtype: list
        """

    @abc.abstractmethod
    def get_step_nodes(self, step_id):
        """Return the compute nodes of a specific job or allocation
           in a list with the duplicates removed.

        :param job_id: job step id or allocation id
        :type job_id: str
        :raises LauncherError: if allocation or job step cannot be
                               found
        :return: list of compute nodes the job was launched on
        :rtype: list of str
        """

    @abc.abstractmethod
    def run(self, step):
        """Run a job step through a launcher

        :param step: Step instance
        :type step: Step
        :return: job_step id
        :rtype: str
        """

    @abc.abstractmethod
    def stop(self, step_id):
        """Stop a job step

        :param step_id: id of the step to be stopped
        :type step_id: str
        :raises LauncherError: if unable to stop job step
        :return: a StepInfo instance
        :rtype: StepInfo
        """
