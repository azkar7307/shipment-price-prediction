import logging
import os, sys
from datetime import datetime

LOG_DIR = "logs" # Folder / Directory

LOG_DIR = os.path.join(os.getcwd(), LOG_DIR)

os.makedirs(LOG_DIR, exist_ok = True)

#log/2023-10-13:19:35.log

CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

file_name = f"log_{CURRENT_TIME_STAMP}.log" # File

log_file_path = os.path.join(LOG_DIR, file_name)

logging.basicConfig(filename = log_file_path,
                    filemode='w',
                    format = '%(asctime)s %(levelname)s %(name)s %(message)s',
                    level = logging.INFO)

# logging.basicConfig(filename="listassignment.log",level=logging.DEBUG,format='%(asctime)s %(levelname)s %(name)s %(message)s')