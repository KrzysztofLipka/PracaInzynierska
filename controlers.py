from flask import Flask, render_template, make_response , jsonify, request, url_for, json
from  models import ingredientSearch, cleaning_text,entity_rec
import requests
app = Flask(__name__)

@app.route('/updateEditorState')
def update():
    inputText = request.args.get('text', 0, type=str)
    ingridients =  ingredientSearch(inputText)
    cleaning_text(inputText)
    return jsonify(ingridients = ingridients)

@app.route('/namedEntity')
def namedEntity():
    inputText = request.args.get('text', 0, type=str)
    ent = entity_rec(inputText)
    return jsonify(entities=ent)


@app.route("/")
def index():
    return render_template('index.html')



if __name__ == "__main__":
    app.run()