# main.py
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "AI Supply Chain Assistant Flask Backend is Running!"})

if __name__ == '__main__':
    app.run(debug=True)
