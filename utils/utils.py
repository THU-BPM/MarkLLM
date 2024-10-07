# Copyright 2024 THU-BPM MarkLLM.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ==============================
# utils.py
# Description: Utility functions
# ==============================

import os
import json


def load_config_file(path: str) -> dict:
    """Load a JSON configuration file from the specified path and return it as a dictionary."""
    try:
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return config_dict

    except FileNotFoundError:
        print(f"Error: The file '{path}' does not exist.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in '{path}': {e}")
        # Handle other potential JSON decoding errors here
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # Handle other unexpected errors here
        return None


def load_json_as_list(input_file: str) -> list:
    """Load a JSON file as a list of dictionaries."""
    res = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
    for line in lines:
        d = json.loads(line)
        res.append(d)
    return res


def create_directory_for_file(file_path) -> None:
    """Create the directory for the specified file path if it does not already exist."""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
