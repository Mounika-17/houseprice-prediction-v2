import logging
import os
from datetime import datetime

# Creates a log file name with a timestamp, like: 2025-09-16_14-22-05.log
LOG_FILE=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
# It creates a folder like C:\Mounika\All Projects\logs\2025-09-16_14-22-05.log
logs_path=os.path.join(os.getcwd(),"logs",LOG_FILE)
# Creates directories recursively.
os.makedirs(logs_path,exist_ok=True)
# Now it joins again → logs_path (already has the file name) + LOG_FILE.So it becomes: C:\Mounika\All Projects\logs\2025-09-16_14-22-05.log\2025-09-16_14-22-05.log
LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)