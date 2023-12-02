import tkinter as tk
from tkinter import scrolledtext
import sys
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from ttkthemes import ThemedTk
from PIL import Image, ImageTk

# Cargar el modelo previamente entrenado
model = load_model('chatbot_model.h5')

# Resto del código de preprocesamiento y carga de datos...
lemmatizer = WordNetLemmatizer()

# Importamos los archivos generados en el código anterior
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Pasamos las palabras de oración a su forma raíz
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Convertimos la información a unos y ceros según si están presentes en los patrones
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predecimos la categoría a la que pertenece la oración
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    max_index = np.where(res == np.max(res))[0][0]
    category = classes[max_index]
    return category

# Obtenemos una respuesta aleatoria
def get_response(tag, intents_json):
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i['responses'])
            break
    return result

# Función para manejar la entrada del usuario y mostrar la respuesta del chatbot
def send_message():
    user_message = user_input.get()
    user_input.delete(0, tk.END)  # Borra el campo de entrada del usuario

    if user_message.lower() == "salir":
        chat_area.insert(tk.END, "Tú: " + user_message + "\n")
        chat_area.insert(tk.END, "Bot: Adiós. Hasta luego.\n")
        chat_area.yview(tk.END)
        sys.exit()  # Salir del programa si el usuario escribe "salir"

    # Realiza la lógica de procesamiento del mensaje del usuario y obtén la respuesta del chatbot
    ints = predict_class(user_message)
    bot_response = get_response(ints, intents)

    chat_area.insert(tk.END, "Tú: " + user_message + "\n")
    chat_area.insert(tk.END, "Bot: " + bot_response + "\n")
    chat_area.yview(tk.END)

# Configuración de la ventana principal con tema avanzado
window = ThemedTk(theme="arc")  # Puedes elegir un tema diferente

# Área de chat con fondo y barra de desplazamiento
chat_area = scrolledtext.ScrolledText(window, wrap=tk.WORD, width=50, height=20)
chat_area.pack(pady=10)
chat_area.configure(bg='#F5F5F5')  # Cambia el color de fondo del área de chat

# Campo de entrada del usuario con estilo
user_input = tk.Entry(window, width=40, font=("Helvetica", 14))
user_input.pack(pady=10, padx=10)
user_input.focus()  # Establece el enfoque en el campo de entrada al iniciar la aplicación

# Botón para enviar mensaje con estilo
send_button = tk.Button(window, text="Enviar", font=("Helvetica", 12), command=send_message)
send_button.pack(pady=5)

# Botón para salir con estilo
exit_button = tk.Button(window, text="Salir", font=("Helvetica", 12), command=sys.exit)
exit_button.pack(pady=5)

# Imagen de avatar para el bot
bot_image = Image.open('docs.png')
bot_image = ImageTk.PhotoImage(bot_image)

# Imagen de avatar para el usuario
user_image = Image.open('usuario.png')
user_image = ImageTk.PhotoImage(user_image)

# Etiquetas para los avatares del bot y el usuario
bot_avatar_label = tk.Label(window, image=bot_image)
bot_avatar_label.pack(side=tk.LEFT, padx=10)

user_avatar_label = tk.Label(window, image=user_image)
user_avatar_label.pack(side=tk.RIGHT, padx=10)

# Mostrar una bienvenida inicial
initial_message = "Bot: ¡Hola! ¿En qué puedo ayudarte hoy?"
chat_area.insert(tk.END, initial_message + "\n")

# Iniciar el bucle principal de la GUI
window.mainloop()
