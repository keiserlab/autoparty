import os
from  decouple import config

class Config(object):

    # Set up the App SECRET_KEY
    SECRET_KEY = config('SECRET_KEY', default='S#perS3crEt_007')

    
    ## change
    CELERY_BROKER_URL = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND = "redis://localhost:6379/0"

    #CELERY_ACCEPT_CONTENT = ["pickle", "json"]

class ProductionConfig(Config):
    DEBUG = False

    # Security
    SESSION_COOKIE_HTTPONLY  = True
    REMEMBER_COOKIE_HTTPONLY = True
    #REMEMBER_COOKIE_DURATION = 3600

    # PostgreSQL database
    SQLALCHEMY_DATABASE_URI = '{}://{}:{}@{}:{}/{}'.format(
        config( 'DB_ENGINE'   , default='mysql+pymysql'    ),
        config( 'DB_USERNAME' , default='hitpicker'       ),
        config( 'DB_PASS'     , default='hpppw'          ),
        config( 'DB_HOST'     , default='127.0.0.1'     ),
        config( 'DB_PORT'     , default=3306        ),
        config( 'DB_NAME'     , default='hpp_db' )
    )

class DebugConfig(Config):
    DEBUG = True

    basedir = os.path.abspath(os.path.dirname(__file__))
    #print("BASEDIR", basedir)
    SQLALCHEMY_DATABASE_URI = 'sqlite:///' + os.path.join(basedir, 'mnt', 'db.sqlite3')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    print("SQLALCHEMY_DATABASE_URI", SQLALCHEMY_DATABASE_URI)


# Load all possible configurations
config_dict = {
    'Production': ProductionConfig,
    'Debug'     : DebugConfig,
    'Test'      : DebugConfig,   
}
