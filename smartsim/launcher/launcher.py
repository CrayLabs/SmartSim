import abc


class Launcher(abc.ABC):
    def __init__(self):
        pass

    @abc.abstractmethod
    def create_step(self, name, cwd, step_settings):
        raise NotImplementedError

    @abc.abstractmethod
    def get_step_update(self, step_names):
        raise NotImplementedError

    @abc.abstractmethod
    def get_step_nodes(self, step_names):
        raise NotImplementedError

    @abc.abstractmethod
    def run(self, step):
        raise NotImplementedError

    @abc.abstractmethod
    def stop(self, step_name):
        raise NotImplementedError
