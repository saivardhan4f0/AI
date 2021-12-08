import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def first():
    return render_template('index.html')

@app.route('/predict', methods=['post'])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction


    return render_template('index.html', pred='the predicted crop is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

