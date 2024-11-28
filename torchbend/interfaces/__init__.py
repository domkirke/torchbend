import os

def get_interface_dependencies():
    interfaces = {}
    req_path = os.path.join(os.path.dirname(__file__), "requirements")
    for f in os.listdir(req_path):
        if os.path.splitext(f)[1] == ".txt":
            with open(os.path.join(req_path, f), 'r') as pp:
                requirements = pp.read().split('\n')
            interfaces[os.path.splitext(f)[0]] = requirements
    return interfaces

    