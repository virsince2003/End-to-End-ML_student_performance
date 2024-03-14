from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = "-e ."
requires_path = "requirements.txt"
def get_requirements(file_path:str)->List[str]:
    requriements = []
    with open (file_path, "r") as f:
        requriements = [object.strip() for object in f.readlines() if hyphen_e_dot not in object]
    return requriements

print(get_requirements(requires_path))
setup(    
    name="End-to-End-student-performance",
    version="0.0.1",
    auther="VirenderChauhan",
    author_email="virchauhan657@gmail.com",
    packages=find_packages(),
    install_requirements=get_requirements(requires_path)
    )
