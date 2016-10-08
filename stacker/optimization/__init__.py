from logging.config import fileConfig
from os import path

log_file_path = path.join(path.dirname(path.abspath(__file__)), 'logging_config.ini')
fileConfig(log_file_path)
