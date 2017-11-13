from flask import Flask, render_template, make_response , jsonify, request, url_for, json
from  models import ingredientSearch, cleaning_text,entity_rec
import requests
from from_spacy import character_searching
app = Flask(__name__)

#@app.route('/updateEditorState')
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

#@app.route('/characterSearching')
@app.route('/updateEditorState')
def char_searching():
    inputText = request.args.get('text', 0, type=str)
    char_list = character_searching(inputText)
    ingridients = ingredientSearch(inputText)
    return jsonify(characters = char_list, ingridients = ingridients)




@app.route("/")
def index():
    return render_template('index.html')



if __name__ == "__main__":
    app.run()