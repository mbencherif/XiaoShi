import json
import time
from flask import Flask, render_template, request

from model.model import Model

app = Flask(__name__, template_folder='./templates', static_url_path='/static')

model = Model('model_400')

def render_html(name):
    return open('templates/'+name, 'r').read()

@app.route("/")
def index():
    return render_html('index.html')

messages = []

@app.route("/send", methods=["POST"])
def send():
    message = request.json
    digits = "零一二三四五六七八九"
    modern = ''.join([digits[int(ch)] if str.isdigit(ch) else ch for ch in message['modern']])
    ancient = model.predict(modern)
    message['ancient'] = ancient

    return json.dumps(message)

if __name__ == '__main__':
    app.run(host='0.0.0.0')