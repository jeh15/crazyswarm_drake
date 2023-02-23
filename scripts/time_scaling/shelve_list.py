from typing import List, Dict
import shelve

def shelve_list(filename: str, key_list: List, workspace_variable_names: Dict, local_variables: Dict):
    shelf = shelve.open(filename, 'n')
    for key in workspace_variable_names:
        if key in key_list:
            shelf[key] = local_variables[key]

    shelf.close()