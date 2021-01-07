import os

from .entity import SmartSimEntity


class DBNode(SmartSimEntity):
    """DBNode objects are the entities that make up the orchestrator.
    Each database node can be launched in a cluster configuration
    and take launch multiple databases per node.

    To configure how each instance of the database operates, look
    into the smartsimdb.conf.
    """

    def __init__(self, name, path, run_settings, ports):
        """Initialize a database node within an orchestrator.

        :param name: identifier for dbnode
        :type name: str
        :param path: path to output and error files
        :type path: str
        :param run_settings: how dbnode should be run, set by orchestrator
        :type run_settings: dict
        :param ports: list of int ports for the dbnode (multiple if dpn > 1)
        :type port: list

        """
        super().__init__(name, path, run_settings)
        self.ports = ports

    def _get_db_conf_filename(self, port):
        """Returns the .conf file name for the given port number

        :param port: port number
        :type port: int
        :return: the dbnode configuration file name
        :rtype: str
        """
        return " ".join(("nodes-", self.name, "-", str(port), ".conf"))

    def remove_stale_dbnode_files(self):
        """This function removes the .conf, .err, and .out files that
        have the same names used by this dbnode that may have been
        created from a previous experiment execution.
        """

        for port in self.ports:
            conf_file = "/".join((self.path, self._get_db_conf_filename(port)))
            if os.path.exists(conf_file):
                os.remove(conf_file)

        if "out_file" in self.run_settings:
            out_file = self.run_settings["out_file"]
            if os.path.exists(out_file):
                os.remove(out_file)

        if "err_file" in self.run_settings:
            err_file = self.run_settings["err_file"]
            if os.path.exists(err_file):
                os.remove(err_file)
