# -*- encoding: utf-8 -*-

from flask_wtf import FlaskForm
from flask_wtf.file import FileField
from wtforms import StringField, PasswordField, RadioField, SelectField, IntegerField, BooleanField, SubmitField
from wtforms.validators import Required, InputRequired, DataRequired

## login and registration

class LoginForm(FlaskForm):
    username = StringField('Username', id='username_login', validators=[DataRequired()])
    password = PasswordField('Password', id='pwd_login', validators=[DataRequired()])

class CreateAccountForm(FlaskForm):
    username = StringField('Username', id='username_create', validators=[DataRequired()])
    password = PasswordField('Password', id='pwd_create', validators=[DataRequired()])
    
class ScreenForm(FlaskForm):
    runname = StringField('RunName', id='p_name')
    existing_screens = SelectField("Append to Existing Screen", id='prevscreens', choices = [], default = 0)
    chunk_csv = StringField('ChunkCSV', id='chunk_csv', validators=[DataRequired()])
    receptor_pdb = StringField('RecPDB', id='rec_pdb', validators=[DataRequired()])
    chunk_directory = StringField('ResDir', id='results_dir', validators=[DataRequired()])
    
class GradeForm(FlaskForm):
    choice_switcher = RadioField('Grade', id='grade', choices=[])

class MethodForm(FlaskForm):
    method_switcher = SelectField("Method to load grades", id="methodswitcher", 
                                    choices = [("score", "Score"), ("uncertainty", "Uncertainty"), ("prediction", "Prediction"), ("disagreement", "Disagreement")], 
                                    default = 'score')
    mode_switcher = SelectField("Mode to load grades", id="modeswitcher", 
                                    choices = [("annotate", "Annotate"), ("review", "Review")], 
                                    default = 'annotate')

class HitpickerRunForm(FlaskForm):
    
    # run settings
    preloaded_id = SelectField('Docking Screen', id = 'run', choices = [])
    party_config = FileField("Party Configuration File", id="party_config")

    req_interactions =  StringField('Required Interactions', id='req_it', default = '')
    max_tc = StringField('Maximum Tanimoto Coefficient', id = 'max_tc', default = '1')
    retrain_iter =  IntegerField('Retrain every:', id='retrain', default = 200)

    # model settings
    hidden_layers = RadioField('# Layers', id='layers', choices = [(2, '2'), (3, '3'), (4, '4'), (5, '5')], default = 2)
    funnel_layers = BooleanField('Funnel Layers', id='funnel', default = "checked")
    #funnel_layers = RadioField('Funnel layers', id='funnel', choices = [(True, 'True'), (False, 'False')], default = True)
    num_neurons = RadioField('# Neurons', id='neurons', choices = [(1024, '1024'), (2048, '2048'), (4096, '4096')], default = 1024)
    learning_rate = RadioField('Learning Rate', id='lr', choices = [(1e-4, '1e-4'), (1e-3, '1e-3'), (1e-2, '1e-2')], default = 1e-4)

    hp_settings_id = SelectField("Existing Parties", id='hpsettingsid', choices = [])
    submit = SubmitField("Submit")
