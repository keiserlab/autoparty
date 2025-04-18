import pytest

import json
from sqlalchemy import inspect

from app import create_app, db
from config import config_dict

from app.base.database import Molecule

@pytest.fixture
def app():
	test_config = config_dict['Test']
	app = create_app(test_config)
	with app.app_context():
		db.create_all() # create all tables
		yield app
		#db.session.remove() # avoids occasional hanging
		#db.drop_all()

def test_database_schema(app):
	"""
	Check that database has been created correctly and has expected columns
	"""
	table_names = inspect(db.engine).get_table_names()
	assert all([ table in table_names for table in  \
		['grades', 'hp_run_settings', 'molecules', 'predictions', 'preloaded', 'user']])

def test_database_insert(app):
	"""
	Insert new molecule into database
	"""
	test_mol = Molecule(name='test_mol',
        run_id=0,
        smi='CCCCCCC',
        score=-1,
        inters=json.dumps({}), #empty interactions
        mol="",
        ifp=None)

	db.session.add(test_mol)
	db.session.commit()

	# assert that molecule was added
	assert Molecule.query.one()

def test_database_retrieval(app):
	assert Molecule.query.one()
	assert Molecule.query.filter_by(run_id = 0).one()
	assert not len(Molecule.query.filter(Molecule.run_id > 0).all()) # should be 0

def test_drop_all(app):
	"""
	This is run last, clears up database and ensure things are gone
	"""
	db.drop_all()

	assert len(inspect(db.engine).get_table_names()) == 0
	try:
		mol = Molecules.query.one()
		assert False # if we make it this far, something has done wrong
	except:
		assert True # retrieval failed, we're good
