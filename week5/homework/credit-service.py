from flask import Flask
from flask import request
from flask import jsonify

import pickle

# Load model
model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

print(dv, model)

app = Flask('score')


@app.route('/credit', methods=['POST'])
def credit_check():
    client = request.get_json()

    X = dv.transform([client])
    y_pred = model.predict_proba(X)[0, 1]

    result = {
        'score_probability': float(round(y_pred, 3)),
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9898)
