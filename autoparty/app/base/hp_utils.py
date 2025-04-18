from sqlalchemy.sql import exists
from sqlalchemy import desc

from app import db #celery, db
from app.base.database import Molecule, Grade, Prediction

import time

def _dump_q_to_list(m, q):
    
    q_list = []
    new_mol_queue = m.Queue()
    
    while not q.empty():
        zid = q.get()
        q_list.append(zid)
        new_mol_queue.put(zid)
    
    return q_list, new_mol_queue

def _still_running(task_name):
    #any_active = celery.control.inspect().active()
    any_active = None
    if any_active:
        active_tasks = any_active.values()
    else:
        return False
    for tasks in active_tasks:
        if any([task['name']==task_name for task in tasks]):
            return True
    return False
    
def populate_queue(q, run_id, user_id, hp_run_id = None, model_id = None, method = 'new', n = 2000):
    
    if method == 'new': # populate by dock score, those that are not already graded by this user
        # TODO: check database query logic
        stmt = exists().where(Molecule.zid == Grade.zid).where(Grade.user_id == user_id)
        molecules = Molecule.query.filter_by(run_id = run_id).filter_by(
            ).filter(~stmt).order_by(Molecule.dock).all()
        for mol in molecules:
            q.append(mol)
            
    elif method == 'top': # populate by dock score, those that are not already graded by this user
        # TODO: check database query logic
        stmt = exists().where(Molecule.zid == Grade.zid).where(Grade.user_id == user_id
                                                              ).where(Grade.hp_run_id == hp_run_id)
        molecules = Molecule.query.filter_by(run_id = run_id).filter_by(
            ).filter(~stmt).order_by(Molecule.dock).all()
        for mol in molecules:
            q.append(mol)
            
    elif method == 'uc': # populate by uncertainty
        """
        Sort by uncertainty, populate with the top 1k by variance
        """
        if model_id is None:
            _,_, model_id = load_last_model(run_id, user_id)
        molecules = db.session.query(Prediction, Molecule).filter(Prediction.zid == Molecule.zid
            ).filter_by(user_id = user_id, model_id = model_id).order_by(desc(Prediction.uncertainty))
                
        
        for mol in molecules:
            q.append(mol[1])
            
    else:
        assert False, "Not yet implemented"
        

def load_last_model(run_id, user_id):
    #pass
    # prev model is most recent model trained by this user
    prev_model = MLModel.query.filter_by(run_id = run_id
                                             ).filter_by(user_id=user_id).order_by(desc(MLModel.timestamp)).first()

    model_params = [prev_model.n_classifiers,
                     prev_model.n_layers,
                     prev_model.neurons,
                     prev_model.learning_rate,
                     'trainall',
                     'singleton',
                     prev_model.input_size,
                     prev_model.labels.split(',')]

    iteration = prev_model.iteration + 1

    c = EnsembleClassifier(*model_params)

    for member, model_state, optim_state in zip(c.classifiers,
                                                    prev_model.model_state, 
                                                    prev_model.optimizer_state):
        member.load_state_dict(model_state)
        member.optim.load_state_dict(optim_state)   
    
    return c, iteration, prev_model.id
