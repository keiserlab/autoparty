
from decouple import config
from config import config_dict

from flask import Flask, url_for
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
#from flask_session import Session
import importlib
from importlib import import_module
from logging import basicConfig, DEBUG, getLogger, StreamHandler
from os import path
from celery import Celery

from config import config, config_dict, Config

#from app.celery_util import make_celery

#from app.base.app_extensions import celery, db

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
celery = Celery(__name__,
    broker = Config.CELERY_BROKER_URL,
    backend = Config.CELERY_RESULT_BACKEND)
SESSION_TYPE = 'redis'

def register_extensions(app):
    db.init_app(app)
    migrate.init_app(app, db)
    login_manager.init_app(app)
    celery.conf.update(app.config)

def register_blueprints(app):
    for module_name in ('base', 'home'):
        module = import_module('app.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)

def configure_database(app):

    @app.before_first_request
    def initialize_database():
        db.create_all()

    @app.teardown_request
    def shutdown_session(exception=None):
        db.session.remove()

def make_celery(celery, app):
    celery.flask_app = app
    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with celery.flask_app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask

def create_app(app_config):

    app = Flask(__name__, static_folder='base/static')
    app.config.from_object(app_config)
    
    register_extensions(app)
    register_blueprints(app)
    configure_database(app)
    make_celery(celery, app)
    return app

