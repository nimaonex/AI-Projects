import json
import sys
from pathlib import Path
import os


#this function returns the full direction of a given file format
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS #PyInstaller executable, is a temporary folder path created by PyInstaller, exists only inside .exe builds.
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

#this functions returns the details of config file
def load_config(config_path="config.json"):
    with open(resource_path(config_path), "r", encoding="utf-8") as config_file:
        return json.load(config_file)