from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager

# Extensions are created here and initialized in app.py

db = SQLAlchemy()
login_manager = LoginManager()
login_manager.login_view = "login"
