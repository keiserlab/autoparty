# -*- encoding: utf-8 -*-
"""
Copyright (c) 2019 - present AppSeed.us
"""


"""
Run:
1 terminal: redis-server
1 terminal: celery worker: celery -A app.celery_util:celery worker --loglevel=DEBUG
"""

from os import environ
from sys import exit, argv
from config import config, config_dict
import logging

from app import create_app, db

DEBUG = config('DEBUG', default=True, cast=bool)

print(DEBUG)
#DEBUG = False

# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'
print("debug bool: ", DEBUG, get_config_mode)

try: 
    # Load the configuration using the default values 
    app_config = config_dict[get_config_mode.capitalize()]
    print("Loaded app config")

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

application = create_app(app_config) 

print(f"In debug - {get_config_mode}")
application.logger.info('DEBUG       = ' + str(DEBUG)      )
application.logger.info('Environment = ' + get_config_mode )
application.logger.info('DBMS        = ' + app_config.SQLALCHEMY_DATABASE_URI )

if __name__ == "__main__":
    application.logger.info('in name == main')
    port = 5000
    if len(argv) > 1:
        try:
            port = int(argv[1])
        except:
            pass
    application.logger.info(f'port: {port}')
    application.run(host = "0.0.0.0", port = port, debug = False)
