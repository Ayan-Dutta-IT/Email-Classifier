from flask import Flask, render_template, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests if needed

# Load vectorizer and model
with open('email_vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

with open('email_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        email = request.form['email']
        input_features = vectorizer.transform([email])
        prediction = model.predict(input_features)[0]
        result = "Ham mail 💌" if prediction == 1 else "Spam mail 🚫"
        return render_template('index.html', prediction=result)


# For API request (optional)
@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    email = data.get('email', '')
    input_features = vectorizer.transform([email])
    prediction = model.predict(input_features)[0]
    result = "Ham mail 💌" if prediction == 1 else "Spam mail 🚫"
    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=False)
