from flask import Flask, request, render_template
from main import predict_requirement

app = Flask(__name__)

history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    req_text = request.form['input_text']
    result = predict_requirement(req_text)
    labels_list = result[0]
    confidence_list = [str(round(float(c) * 100, 2)) for c in result[1]]

    # Add the current prediction and input text to the history
    history.insert(0, (req_text, labels_list, confidence_list))

    # Pass the prediction results and history to the HTML template
    return render_template('index.html', results=list(zip(labels_list, confidence_list)), history=history, req=req_text)

if __name__ == '__main__':
    app.run(debug=True)