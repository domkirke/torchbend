from distutils.core import setup
from setuptools import find_packages
import importlib, os

import os

def get_interface_dependencies():
    interfaces = {}
    req_path = os.path.join(os.path.dirname(__file__), "torchbend", "interfaces", "requirements")
    for f in os.listdir(req_path):
        if os.path.splitext(f)[1] == ".txt":
            with open(os.path.join(req_path, f), 'r') as pp:
                requirements = pp.read().split('\n')
            interfaces[os.path.splitext(f)[0]] = requirements
    return interfaces

def find_interfaces():
      return get_interface_dependencies()
    
with open("requirements.txt", "r") as requirements:
    requirements = requirements.read()

setup(name='torchbend',
      version='0.1',
      description='Machine learning experimental library for model dissection and bending',
      author='Axel Chemla--Romeu-Santos',
      python_requires="==3.11.*",
      packages=find_packages(),
      install_requires=requirements, 
      extras_require=find_interfaces()
     )


# if __name__ == "__main__":
#       print(find_interfaces())