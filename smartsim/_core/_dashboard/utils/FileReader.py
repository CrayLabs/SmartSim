import json
from typing import List, Tuple, Dict, Any, Optional, Union


class ManifestReader:
    def __init__(self, filename: str):
        self.filename = filename
        self.data: Optional[Dict[str, Any]] = None
        self.experiment: Dict[str, str] = {}
        self.runs: List[Dict[str, Any]] = []
        self.applications: List[Dict[str, Any]] = []
        self.orchestrators: List[Dict[str, Any]] = []
        self.ensembles: List[Dict[str, Any]] = []
        self.load_data()

    def load_data(self) -> None:
        try:
            with open(self.filename, "r", encoding="utf-8") as json_file:
                self.data = json.load(json_file)
        except FileNotFoundError:
            self.data = None
            self.experiment = {}
            return

        if self.data is not None:
            self.experiment = self.data.get("experiment", {})
            self.runs = self.data.get("runs", [])
            self.applications = [
                app for run in self.runs for app in run.get("applications", [])
            ]
            self.orchestrators = [
                orch for run in self.runs for orch in run.get("orchestrators", [])
            ]
            self.ensembles = [
                ensemble for run in self.runs for ensemble in run.get("ensembles", [])
            ]

    def get_entity(
        self, entity_name: str, entity_list: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        This function gets an entity from a list of entities when you only
        know its name.

        Args:
            entity_name (str): The name of the entity to search for. This is
            gotten from a dropdown in the dashboard.
            entity_list (List[Dict[str, Any]]): The list of entites
            to search through.

        Returns:
            Optional[Dict[str, Any]]: If found, returns the entity.
            Otherwise returns None.
        """
        if self.data is None:
            return None
        for entity in entity_list:
            if entity and "name" in entity and entity["name"] == entity_name:
                return entity
        return None

    def get_entity_value(self, key: str, entity: Optional[Dict[str, Any]]) -> Any:
        """
        This function gets an attribute from an entity. This function does
        not return None because the dashboard should show an empty
        field if the value doesn't exist.

        Args:
            key (str): The key to get the corresponding value from.
            entity (Optional[Dict[str, Any]]): The entity that you want the value from.

        Returns:
            Any: If found, returns the value. This could be practically anything.
            If not found, depending on what type the dashboard needs to show,
            it returns that type without anything in it.
        """
        if entity and self.data is not None:
            value = entity.get(key, "")
            if key == "interface" and isinstance(value, List):
                return ", ".join(value)
            return value
        if key in ("exe_args", "db_hosts"):
            return []
        if key == "colocated_db_settings":
            return {}
        return ""

    def get_entity_dict_keys_and_values(
        self, dict_name: str, entity: Optional[Dict[str, Any]]
    ) -> Tuple[List[str], List[str]]:
        """
        This function properly formats the keys and values of a dictionary
        based on how the information is displayed in the dashboard.

        Args:
            dict_name (str): The name of the dict that need to be formatted.
            entity (Optional[Dict[str, Any]]): The entity we get the dict from.

        Returns:
            Tuple[List[str], List[str]]: Returns a tuple of keys and values
            to be displayed in the dashboard.
        """
        keys = []
        values = []

        if entity and self.data is not None:
            target_dict = entity.get(dict_name, {})
            for key, value in target_dict.items():
                if isinstance(value, List) and dict_name == "params":
                    comma_separated_string = ", ".join(value)
                    keys.append(key)
                    values.append(comma_separated_string)
                elif isinstance(value, List):
                    for v in value:
                        keys.append(key)
                        values.append(str(v))
                elif isinstance(value, Dict):
                    for k, v in value.items():
                        keys.append(k)
                        values.append(str(v))
                else:
                    keys.append(key)
                    values.append(str(value))

        return keys, values

    def get_loaded_entities(
        self, entity: Optional[Dict[str, Any]]
    ) -> Union[List[Dict[str, str]], Dict[str, List[Any]]]:
        """
        This function properly combines and formats the keys and values of
        DB Models and DB Scripts so they can be displayed as "Loaded Entities"
        in the dashboard.

        Args:
            entity (Optional[Dict[str, Any]]): The entity we get DB Models
            and Db Scripts from.

        Returns:
            Union[List[Dict[str,str]], Dict[str, List[Any]]]: Returns a list of dicts
            with Name, Type, Backend, and Device as the keys. If there are no DB
            Models or DB Scripts, or the entity passed in doesn't exist, this function
            returns a single dict with the headers for the table and empty lists as
            their values. The dashboard displays that there is no data when this is
            done.
        """
        loaded_data = []
        if entity and self.data is not None:
            for item in entity.get("db_models", []):
                for key, value in item.items():
                    loaded_data.append(
                        {
                            "Name": key,
                            "Type": "DB Model",
                            "Backend": value["backend"],
                            "Device": value["device"],
                        }
                    )
            for item in entity.get("db_scripts", []):
                for key, value in item.items():
                    loaded_data.append(
                        {
                            "Name": key,
                            "Type": "DB Script",
                            "Backend": value["backend"],
                            "Device": value["device"],
                        }
                    )

            if not loaded_data:
                return {"Name": [], "Type": [], "Backend": [], "Device": []}
            return loaded_data

        return {"Name": [], "Type": [], "Backend": [], "Device": []}

    # Ensemble Members
    def get_ensemble_members(
        self, ensemble: Optional[Dict[str, Any]]
    ) -> List[Optional[Dict[str, Any]]]:
        """
        This function gets all of the members inside of an ensemble.

        Args:
            ensemble (Optional[Dict[str, Any]]): Ensemble to get its members from.

        Returns:
            Optional[Dict[str, Any]]: Returns a list of ensemble members,
            or an empty list.
        """
        if ensemble and self.data is not None:
            return ensemble.get("members", [])
        return []

    def get_member(
        self, member_name: str, ensemble: Optional[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        This function gets a specific member form an ensemble

        Args:
            member_name (str): Nmae of the member to be returned. This comes
            from a dashboard dropdown.
            ensemble (Optional[Dict[str, Any]]): Ensemble to get its members from.

        Returns:
            Optional[Dict[str, Any]]: Returns a list of ensemble members,
            or an empty list.
        """
        for member in self.get_ensemble_members(ensemble):
            if member and "name" in member and member["name"] == member_name:
                return member
        return None
