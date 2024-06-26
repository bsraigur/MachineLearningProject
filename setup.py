from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = "-e ."  # in requirements.txt used to activate\run setup.py
def get_requirements(file_path:str)->List[str]:
    '''
    this function return the list of requirements
    '''
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]
    
    if HYPEN_E_DOT in requirements:
        requirements.remove(HYPEN_E_DOT)
    
    return requirements


setup(
    name="MachineLearningProject",
    version="1.0",
    author="BharatSingh Rajpurohit",
    author_email="bsraigur@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)