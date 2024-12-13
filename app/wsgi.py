
from os import environ
from sys import exit, argv
from config import config, config_dict
import logging
import atexit
from app import create_app

DEBUG = True

get_config_mode = 'Debug' if DEBUG else 'Production'

app_config = config_dict[get_config_mode.capitalize()]
application = create_app(app_config) 

