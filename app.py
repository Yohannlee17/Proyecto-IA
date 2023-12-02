from flask import Flask, request, jsonify
import random
import json
import pickle
import numpy as np
import nltk
from flask import Flask, render_template,request, jsonify
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

app = Flask(__name__)

# Cargando el modelo y los datos necesarios
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json', encoding='utf-8').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Funciones de preprocesamiento y predicción
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    return result

@app.route("/")
def index():
    return render_template('index.html')  # Asegúrate de que index.html esté en la carpeta 'static'

# Ruta para el chat
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    tag = predict_class(user_input)
    response = get_response(tag, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
