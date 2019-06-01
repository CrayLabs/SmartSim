

from os import  getcwd
import toml

def read_config():
    try:
        cwd = getcwd()
        fname = cwd + "/mpo-config.toml"
        with open(fname, 'r', encoding='utf-8') as fp:
            parsed_toml = toml.load(fp)
            return parsed_toml
    except Exception as e:
        raise Exception("Could not parse/find mpo-config.toml")
