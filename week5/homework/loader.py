import pickle

model_file = 'model_C=1.0.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

model = model
dv = dv

client = {
    "reports": 0,
    "share": 0.001694,
    "expenditure": 0.12,
    "owner": "yes"
}

X = dv.transform([client])

prob = model.predict_proba(X)[0, 1]
print(f'Probability that client will get a credit card is: {round(prob, 3)}')
