from setuptools import find_packages, setup
from typing import List

random_letter = '-e .'

def get_requirements(file_path:str)->List[str]:
    '''
    This function returs the requirements from requirements.txt file
    '''
    # blank requirements list
    requirements = []

    # Opening the file
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        # get rid of -e. if present in requirements
        if random_letter in requirements:
            requirements.remove(random_letter)

    return requirements

setup(
    name = 'End to End Regression ML Project',
    version= '0.0.1',
    author='Manuel Contreras',
    author_email='manuel174102@hotmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)