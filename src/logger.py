import os
import json
from functools import reduce
import operator


def keys_exists(element, keys):
    """
    Check if *keys (nested) exists in `element` (dict).
    """
    if not isinstance(element, dict):
        raise AttributeError("keys_exists() expects dict as first argument.")
    if len(keys) == 0:
        raise AttributeError("keys_exists() expects at least two arguments, one given.")

    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def getFromDict(dataDict, mapList):
    return reduce(operator.getitem, mapList, dataDict)


def setInDict(dataDict, mapList, value):
    getFromDict(dataDict, mapList[:-1])[mapList[-1]] = value


class Logger:
    def __init__(self, folder, filename, enable=False):
        self.enable = enable
        if self.enable:
            if not os.path.exists(folder):
                os.makedirs(folder)

            self.path = os.path.join(folder, f"{filename}.json")

    def create(self):
        if self.enable:
            with open(self.path, "w") as hundle:
                json.dump({}, hundle, indent=2)

    def update(self, key, content):
        if self.enable:
            with open(self.path, "r") as hundle:
                json_file = json.load(hundle)

            if isinstance(key, str):
                json_file[key] = content
            elif hasattr(key, "__iter__"):
                for i in range(len(key) - 1):
                    if not keys_exists(json_file, key[: i + 1]):
                        setInDict(json_file, key[: i + 1], {})
                setInDict(json_file, key, content)
            else:
                raise Exception("key must be either str or iterable.")

            with open(self.path, "w") as hundle:
                json.dump(json_file, hundle, indent=2)

    def get_values(self, key):
        if self.enable:
            with open(self.path, "r") as hundle:
                json_file = json.load(hundle)
            return json_file[key]
        else:
            return []
