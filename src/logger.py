import logging
import os,pathlib
from datetime import datetime

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent #e:\CampusX\projects\PW\diamond_price_pred

pathlib.Path(home_dir.as_posix()+'/logs').mkdir(parents=True,exist_ok=True)
#logs_path = os.path.join(os.getcwd(), 'logs', LOG_FILE)
#os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = home_dir.as_posix()+'/logs/'+LOG_FILE
#LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
