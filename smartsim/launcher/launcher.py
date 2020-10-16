import abc

class Launcher(abc.ABC):

    def __init__(self, *args, **kwargs):
        super().__init__()

    #-------------- Abstract Methods --------------
    # currently not used
    @abc.abstractmethod
    def validate(self, nodes=None, ppn=None, partition=None):
        """Validate the functionality of the launcher and availability of
           resources on the system

        :param nodes: Override the number of nodes to validate
        :param ppn: Override the processes per node to validate
        :param partition: Override the partition to validate
        :return:
        """
        pass

    @abc.abstractmethod
    def create_step(self, name, run_settings, multi_prog=False):
        """Convert a smartsim entity run_settings into a job step
           to be launched on an allocation. An entity must have an
           allocation assigned to it in the running settings or
           create_step will throw a LauncherError

        :param name: name of the step to be launch, usually entity.name
        :type name: str
        :param run_settings: smartsim run_settings for an entity
        :type run_settings: dict
        :param multi_prog: create a multi-program step, defaults to False
        :type multi_prog: bool, optional
        :raises LauncherError: if step creation fails
        :return: Step object
        """
        pass

    @abc.abstractmethod
    def get_step_status(self, step_id):
        """Return the status of a job step from either the OS or
           the workload manager.

        :param step_id: id of the step in the form of xxxxxx.x
        :type step_id: str
        :return: status of the job step and returncode
        :rtype: tuple of (str, str)
        """
        pass

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
        pass


    @abc.abstractmethod
    def run(self, step):
        """Run a job step on an allocation through the workload manager
           A constructed job step is required such that the argument
           translation from SmartSimEntity to Step has been completed
           and an allocation has been assigned to the step.

        :param step: Step instance
        :type step: Step
        :raises LauncherError: If the allocation cannot be found or the
                               job step failed to launch.
        :return: job_step id
        :rtype: str
        """
        pass

    @abc.abstractmethod
    def stop(self, step_id):
        """Stop a job step within an allocation.

        :param step_id: id of the step to be stopped
        :type step_id: str
        :raises LauncherError: if unable to stop job step
        """
        pass

    @abc.abstractmethod
    def is_finished(self, status):
        """Based on the statuses gleaned from the workload manager
           determine wether a job is finished or not.

        :param status: status parsed from the wlm
        :type status: str
        :returns: True/False wether job is finished
        :rtype: bool
        """
        pass


    @abc.abstractmethod
    def __str__(self):
        pass