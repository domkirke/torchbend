import random
from flask import Flask
from flask_bootstrap import Bootstrap
from flask import Blueprint
from flask_sqlalchemy import SQLAlchemy
main = Blueprint('main', __name__)

import sys
import os, os.path as path
_DEFAULT_CONFIG_PATH = os.environ.get('TORCHBEND_SERVER_PATH', os.path.join(os.path.dirname(__file__), "..", "..", ".server/config.json"))

def _get_rnd_hash(n=8):
    return "".join([chr(random.randrange(97, 122)) for _ in range(n)])

from . import db
from . import config

def create_server(config_file: str =_DEFAULT_CONFIG_PATH):

    app = Flask(
        "torchbend", 
        template_folder=path.join(path.dirname(__file__), "templates"),
        static_folder=path.join(path.dirname(__file__), "static")
    )
    app.config['SECRET_KEY'] = _get_rnd_hash(16)
    app.config['CONFIG_FILE'] = config_file 
    app.register_blueprint(main)

    # look for config file
    config_file = config_file or _DEFAULT_CONFIG_PATH
    if not os.path.exists(config_file):
        config._init_config_file(config_file)
        

    # setup database
    basedir = os.path.abspath(os.path.dirname(__file__))
    app.config['SQLALCHEMY_DATABASE_URI'] =\
        'sqlite:///' + os.path.join(basedir, 'data.sqlite')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    db.db.init_app(app)

    bootstrap = Bootstrap(app)

    return app
    

def get_template_path(template_path):
    return path.join(path.dirname(__file__), "templates", template_path)

from . import server
