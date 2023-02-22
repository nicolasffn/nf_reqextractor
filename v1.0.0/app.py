from flask import Flask, request, render_template
from main import predict_requirement

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    text = request.form['text']
    results = predict_requirement(text)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
