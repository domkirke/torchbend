import os
from flask_sqlalchemy import SQLAlchemy


db = SQLAlchemy()

class ModelType(db.Model):
    __tablename__ = "model_types"
    id = db.Column(db.Integer, primary_key=True)
    media_type = db.Column(db.String)
    models = db.relationship('Model', backref="model_type", lazy="dynamic")

class Model(db.Model):
    __tablename__ = "models"
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), unique=True)
    type_id = db.Column(db.Integer, db.ForeignKey('model_types.id'))

    def __repr__(self):
        return '<Role %r>' % self.name