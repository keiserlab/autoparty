from flask_login import UserMixin, current_user
from sqlalchemy import LargeBinary, Column, Integer, String, Float, ForeignKey
from sqlalchemy.types import DateTime, PickleType, JSON, Boolean
#from sqlalchemy.dialects.mysql import MEDIUMTEXT,LONGTEXT
from sqlalchemy.orm import relationship
from sqlalchemy.inspection import inspect

from app import db, login_manager

from app.base.util import hash_pass

import datetime
import pickle

class Serializer(object):

    def serialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys()}

    @staticmethod
    def serialize_list(l):
        return [m.serialize() for m in l]

def try_unpickle(x):
    try:
        return pickle.loads(x)
    except:
        return x

class User(db.Model, UserMixin):

    __tablename__ = 'user'

    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True)
    password = Column(LargeBinary)
    
    runs = relationship('Preloaded', back_populates='user')
    #mlmodels = relationship('MLModel', back_populates='user')
    uncertains = relationship('Prediction', back_populates='user')
    hp_settings = relationship('HPRunSetting', back_populates='user')

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                value = value[0]

            if property == 'password':
                value = hash_pass( value ) # we need bytes here (not plain str)
                
            setattr(self, property, value)

    def __repr__(self):
        return str(self.username)

class Preloaded(db.Model):
    
    __tablename__ = "preloaded"
    
    id = Column(Integer, primary_key = True, index = True)
    runname = Column(String(50))
    pdbcontents = Column(PickleType)
    luna_config = Column(JSON)
    timestamp = Column(DateTime)
    
    molecules = relationship('Molecule', back_populates='preloaded')
    #mlmodels = relationship('MLModel', back_populates='preloaded')
    grades = relationship('Grade', back_populates='preloaded')
    hp_settings = relationship('HPRunSetting', back_populates='preloaded')
    
    user_id = Column(Integer, ForeignKey(User.id))
    user = relationship("User")
    
    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                value = value[0]
            setattr(self, property, value)
        setattr(self, 'timestamp', datetime.datetime.now())  
        
class Molecule(db.Model, Serializer):
    
    __tablename__ = "molecules"
    
    id = Column(Integer, primary_key = True, index = True)
    name = Column(String(50))
    
    run_id = Column(Integer, ForeignKey(Preloaded.id))
    preloaded = relationship("Preloaded")

    smi = Column(PickleType) # pickled smiles string
    mol = Column(PickleType) # pickled molblock
    score = Column(Float)
    ifp = Column(PickleType)
    inters = Column(JSON)
    meta = Column(JSON)
    
    annotations = relationship('Grade', lazy="dynamic")
    uncertains = relationship('Prediction', lazy="dynamic")
    
    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)

    def serialize(self):
        return {c: try_unpickle(getattr(self, c)) for c in inspect(self).attrs.keys() 
        if c not in ['ifp', 'preloaded', 'annotations', 'uncertains']}

        
class HPRunSetting(db.Model):
    
    __tablename__ = "hp_run_settings"
    
    id = Column(Integer, primary_key = True, index = True)

    preloaded_id = Column(Integer, ForeignKey(Preloaded.id))
    preloaded = relationship("Preloaded")
    
    user_id = Column(Integer, ForeignKey(User.id))
    user = relationship("User")

    grades = relationship("Grade")
    timestamp = Column(DateTime)

    party_config = Column(JSON)

    # to store history of model training over time
    history = Column(JSON)
    
    # model settings
    hidden_layers = Column(Integer)
    funnel = Column(Boolean)
    num_neurons = Column(Integer)
    learning_rate = Column(Float)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            if hasattr(value, '__iter__') and not isinstance(value, str):
                value = value[0]
                
            setattr(self, property, value)
        setattr(self, 'timestamp', datetime.datetime.now())  

            
class Grade(db.Model, Serializer):
    
    __tablename__ = "grades"
    
    id = Column(Integer, primary_key = True, index = True)
    
    run_id = Column(Integer, ForeignKey(Preloaded.id))
    preloaded = relationship("Preloaded")
    
    user_id = Column(Integer, ForeignKey(User.id))
    user = relationship("User")
    
    hp_settings_id = Column(Integer, ForeignKey(HPRunSetting.id))
    hp_settings = relationship("HPRunSetting")
    
    grade = Column(String(1))
    ifp = Column(JSON)
    timestamp = Column(DateTime)
    
    mol_id = Column(Integer, ForeignKey(Molecule.id))
    molecule = relationship("Molecule")
        
    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, 'timestamp', datetime.datetime.now())  

    def serialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys() 
            if c not in ['ifp', 'preloaded', 'timestamp', 'uncertains', 'molecule', 'hp_settings', 'user']}          
    

class Model(db.Model):
    
    __tablename__ = "models"
    
    id = Column(Integer, primary_key = True, index = True)
    run_id = Column(Integer, ForeignKey(Preloaded.id))
    preloaded = relationship("Preloaded")
    
    user_id = Column(Integer, ForeignKey(User.id))
    user = relationship("User")

    hp_run_id = Column(Integer, ForeignKey(HPRunSetting.id))
    hp_run = relationship("HPRunSetting") 
    
    train_loss = Column(Float)
    val_loss = Column(Float)

    num_grades = Column(Integer)
    accuracy = Column(Float)

    timestamp = Column(DateTime)

    def __init__(self, **kwargs):
        for property, value in kwargs.items():
            setattr(self, property, value)
        setattr(self, 'timestamp', datetime.datetime.now())  

class Prediction(db.Model, Serializer):
    
    __tablename__ = "predictions"
    
    mol_id = Column(Integer, ForeignKey(Molecule.id), primary_key = True)
    molecule = relationship("Molecule")
    
    user_id = Column(Integer, ForeignKey(User.id), primary_key = True)
    user = relationship("User")

    hp_settings_id = Column(Integer, ForeignKey(HPRunSetting.id), primary_key = True)
    hp_settings = relationship("HPRunSetting")

    #model_id = Column(Integer, ForeignKey(MLModel.id), primary_key = True)
    #models = relationship("MLModel")
    
    prediction = Column(String(1))
    uncertainty = Column(Float) 
    error = Column(Float) # used to record how far off we are from the assigned grade

    def serialize(self):
        return {c: getattr(self, c) for c in inspect(self).attrs.keys() 
            if c not in ['molecule', 'user', 'hp_settings']}          
    

@login_manager.user_loader
def user_loader(id):
    return User.query.filter_by(id=id).first()

@login_manager.request_loader
def request_loader(request):
    username = request.form.get('username')
    user = User.query.filter_by(username=username).first()
    return user if user else None