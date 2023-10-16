import os, sys
from pathlib import Path
import logging

while True:
    project_name = input("Enter your project name: ") # src
    if project_name != "":
        break

#src/__init__.py
#src/components/__init__.py
#

list_of_files = [
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/constant/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/utils/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"config/config.yaml",
    f"config/schema.yaml",
    "schema.yaml",
    "setup.py",
    "main.py",
    "app.py",
    "logs.py",
    "exception.py",
    "requriments.txt",
    "dvc.yaml"

]

# Folder & Files

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath) # ('src\\components', '__init__.py'), ('', 'setup.py')
    
    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
# 

    if (not os.path.exists(filepath) or (os.path.getsize(filepath) == 0)):
        with open(filepath, "w") as f:
            pass
    else:
        logging.info(f"file is already present at {filepath}")
        