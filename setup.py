from setuptools import find_packages, setup
from typing import List

REQUIRMENTS_FILE_NAME = "requirements.txt"
HYPHEN_E_DOT = "-e ."

def get_requirements()->List[str]:
    with open(REQUIRMENTS_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
    requirement_list = [requirement_name.replace("\n", "") for requirement_name in requirement_list]

    if HYPHEN_E_DOT in requirement_list:
        requirement_list.remove(HYPHEN_E_DOT)

    return requirement_list


setup(name = "shipment-price-prediction",
      version = "0.0.1",
      descriptions = "Data Science projects",
      author = "[Success_Analytics, Mohammad Azkar]",
      authod_email = "[Success_Analytics@gmail.com, azkar7307@gmail.com]",
      packages = find_packages(),
      install_requires =get_requirements() ,
          )