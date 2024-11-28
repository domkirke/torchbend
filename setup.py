from distutils.core import setup
from setuptools import find_packages
import importlib, os

def find_interfaces():
      from torchbend.interfaces import get_interface_dependencies
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