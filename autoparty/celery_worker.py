from app import create_app, make_celery, celery, db
from config import Config, config, config_dict

from app.base.database import Preloaded, Molecule, Grade, Prediction

DEBUG = config('DEBUG', default=True, cast=bool)
#DEBUG = False
# The configuration
get_config_mode = 'Debug' if DEBUG else 'Production'

try: 
    # Load the configuration using the default values 
    app_config = config_dict[get_config_mode.capitalize()]

except KeyError:
    exit('Error: Invalid <config_mode>. Expected values [Debug, Production] ')

app = create_app(app_config)
make_celery(celery, app)
