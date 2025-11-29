"""Path utils."""
from os.path import dirname, abspath


curr_filepath = abspath(__file__)
repo_path = dirname(dirname(dirname(curr_filepath)))


def get_data_root_from_hostname(use_tmp=False):
    import socket
    import os

    # Check use_tmp as an environment variable
    if "USE_TMP" in os.environ:
        use_tmp = True
    
    # TODO: if use_tmp, then use the local /tmp folder on the node

    data_root_lib = {
        "diva": "/ssd/pbagad/datasets/",
        "node": "/var/scratch/pbagad/datasets/",
        "fs4": "/var/scratch/pbagad/datasets/",
        "vggdev21": "/scratch/shared/beegfs/piyush/datasets/",
        "node407": "/var/scratch/pbagad/datasets/",
        "gnodee5": "/scratch/shared/beegfs/piyush/datasets/",
        "gnodeg2": "/scratch/shared/beegfs/piyush/datasets/",
        "gnodec2": "/scratch/shared/beegfs/piyush/datasets/",
        "Piyushs-MacBook-Pro": "/Users/piyush/projects/",
        "dev21": "/scratch/shared/beegfs/piyush/datasets/",
    }
    hostname = socket.gethostname()
    hostname = hostname.split(".")[0]
    
    assert hostname in data_root_lib.keys(), \
        "Hostname {} not in data_root_lib".format(hostname)

    data_root = data_root_lib[hostname]
    return data_root
    
