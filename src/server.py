import json
import time
import os
from flask import Flask, render_template, request

from model.model import Model
from model.config import Path

app = Flask(__name__, template_folder='./templates', static_url_path='/static')

model = Model('model_400')

cur_path = os.path.dirname(os.path.realpath(__file__))

def render_html(name):
    return open(os.path.join(cur_path, 'templates/', name), 'r').read()

def write_log(line):
    with open(os.path.join(Path.logs, 'server', 'logs.txt'), 'w') as f:
        f.write(line)

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

    write_log(request.remote_addr + "," + modern + "," + ancient)

    return json.dumps(message)

if __name__ == '__main__':
    app.run(host='0.0.0.0')