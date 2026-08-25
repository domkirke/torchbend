from flask import render_template, session, redirect, url_for, flash
from . import main as app


from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField 
from wtforms.validators import DataRequired


class NameForm(FlaskForm):
    name = StringField('What is your name?', validators=[DataRequired()]) 
    submit = SubmitField('Submit')


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/<model>", methods=['GET', 'POST'])
def model_main_page(model):
    form = NameForm()
    if form.validate_on_submit(): 
        if session.get('name') is not None and session.get('name') != form.name.data:
            flash('You changed your name to %s'%(form.name.data))
            session['name'] = form.name.data
        session['name'] = form.name.data
        return redirect(url_for('model_main_page', model=model))
    return render_template("model_main.html", model=model, form=form, name=session.get('name'))
