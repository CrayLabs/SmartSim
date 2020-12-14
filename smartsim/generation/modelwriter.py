import re

from ..error import ParameterWriterError
from ..utils import get_logger

logger = get_logger(__name__)


class ModelWriter:
    def __init__(self):
        self.tag = ";"
        self.regex = "(;.+;)"

    def set_tag(self, tag, regex=None):
        """Set the tag for the modelwriter to search for within
           tagged files attached to an entity.

        :param tag: tag for the modelwriter to search for,
                    defaults to semi-colon e.g. ";"
        :type tag: str
        :param regex: full regex for the modelwriter to search for,
                     defaults to "(;.+;)"
        :type regex: str, optional
        """
        if regex:
            self.regex = regex
        else:
            self.tag = tag
            self.regex = "".join(("(", tag, ".+", tag, ")"))

    def configure_tagged_model_files(self, model):
        """Read, write and configure tagged files attached to a Model
           instance.

        :param model: a model instance
        :type model: Model
        """
        logger.debug(f"Configuring model {model.name} with params {model.params}")
        for tagged_file in model.files.tagged:
            self._set_lines(tagged_file)
            self._replace_tags(model)
            self._write_changes(tagged_file)

    def _set_lines(self, file_path):
        """Set the lines for the modelwrtter to iterate over

        :param file_path: path to the newly created and tagged file
        :type file_path: str
        :raises ParameterWriterError: if the newly created file cannot be read
        """
        try:
            fp = open(file_path, "r+")
            self.lines = fp.readlines()
            fp.close()
        except (IOError, OSError):
            raise ParameterWriterError(file_path)

    def _write_changes(self, file_path):
        """Write the ensemble-specific changes

        :raises ParameterWriterError: if the newly created file cannot be read
        """
        try:
            fp = open(file_path, "w+")
            for line in self.lines:
                fp.write(line)
            fp.close()
        except (IOError, OSError):
            raise ParameterWriterError(file_path, read=False)

    def _replace_tags(self, model):
        """Replace the tagged within the tagged file attached to this
           model. The tag defaults to ";"

        :param model: The model instance
        :type model: Model
        """
        edited = []
        unused_tags = {}
        for i, line in enumerate(self.lines):
            search = re.search(self.regex, line)
            if search:
                tagged_line = search.group(0)
                previous_value = self._get_prev_value(tagged_line)
                if self._is_ensemble_spec(tagged_line, model.params):
                    new_val = str(model.params[previous_value])
                    new_line = re.sub(self.regex, new_val, line)
                    edited.append(new_line)

                # if a tag is found but is not in this model's configurations
                # put in placeholder value
                else:
                    tag = tagged_line.split(self.tag)[1]
                    if tag not in unused_tags:
                        unused_tags[tag] = []
                    unused_tags[tag].append(i + 1)
                    edited.append(re.sub(self.regex, previous_value, line))
            else:
                edited.append(line)
        for tag in unused_tags.keys():
            logger.warning(f"Unused tag {tag} on line(s): {str(unused_tags[tag])}")
        self.lines = edited

    def _is_ensemble_spec(self, tagged_line, model_params):
        split_tag = tagged_line.split(self.tag)
        prev_val = split_tag[1]
        if prev_val in model_params.keys():
            return True
        return False

    def _get_prev_value(self, tagged_line):
        split_tag = tagged_line.split(self.tag)
        return split_tag[1]
