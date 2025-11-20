from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from sqlalchemy.orm import relationship
from extensions import db

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    name = db.Column(db.String(120))

    # Preferences
    gender = db.Column(db.String(32))  # male, female, non-binary
    style_preference = db.Column(db.String(64))  # casual, formal, etc.
    favorite_colors = db.Column(db.String(255))  # comma-separated hex codes or names
    height_cm = db.Column(db.Float)
    weight_kg = db.Column(db.Float)
    size = db.Column(db.String(32))
    fashion_goals = db.Column(db.String(255))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    wardrobe_items = relationship('WardrobeItem', back_populates='owner', cascade='all, delete-orphan')

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class WardrobeItem(db.Model):
    __tablename__ = 'wardrobe_items'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)

    # Core attributes
    category = db.Column(db.String(64), nullable=False)  # shirt, pants, shoes, jacket, accessories
    color_hex = db.Column(db.String(7))  # e.g., #RRGGBB
    pattern = db.Column(db.String(64))  # plain, striped, floral, graphic
    material = db.Column(db.String(64))  # cotton, denim, leather, silk
    fit = db.Column(db.String(64))       # slim, oversized, regular
    seasonality = db.Column(db.String(64))  # summer, winter, all-season
    occasion_tags = db.Column(db.String(255))  # comma-separated: work, party, gym, wedding

    # Optional metadata
    brand = db.Column(db.String(128))
    price = db.Column(db.Float)
    size = db.Column(db.String(32))
    image_url = db.Column(db.String(512))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    owner = relationship('User', back_populates='wardrobe_items')

    def occasion_list(self):
        return [o.strip().lower() for o in (self.occasion_tags or '').split(',') if o.strip()]
