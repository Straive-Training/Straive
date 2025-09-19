from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_swagger_ui import get_swaggerui_blueprint
from sklearn.linear_model import LogisticRegression
import numpy as np

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///banking.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# Swagger UI configuration
SWAGGER_URL = '/api/docs'  # URL for exposing Swagger UI (without trailing '/')
API_URL = '/static/swagger.json'  # Path to the swagger.json file

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "Banking Customer API"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)


class Customer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    age = db.Column(db.Integer)
    balance = db.Column(db.Float)


with app.app_context():
    db.create_all()


@app.route('/customer', methods=['POST'])
def add_customer():
    data = request.get_json()
    new_customer = Customer(name=data['name'], age=data['age'], balance=data['balance'])
    db.session.add(new_customer)
    db.session.commit()
    return jsonify({'message': 'Customer added successfully'}), 200


@app.route('/customers', methods=['GET'])
def get_customers():
    customers = Customer.query.all()
    result = [{'id': c.id, 'name': c.name, 'age': c.age, 'balance': c.balance} for c in customers]
    return jsonify(result)


@app.route('/train', methods=['POST'])
def train_model():
    X = np.array([[25, 2000], [40, 6000], [50, 8000], [30, 3000]])
    y = np.array([0, 1, 1, 0])  # 0=low, 1=high income

    global model
    model = LogisticRegression()
    model.fit(X, y)
    return jsonify({'message': 'Model trained successfully'})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array([[data['age'], data['balance']]])
    prediction = model.predict(features)[0]
    return jsonify({'prediction': int(prediction)})

@app.route('/loan_train', methods=['POST'])
def train_loan_model():
    X = np.array([[25, 2000], [40, 6000], [50, 8000], [30, 3000]])
    y = np.array([0, 1, 1, 0])

    global loan_model
    loan_model = LogisticRegression()
    loan_model.fit(X, y)
    return jsonify({'message': 'Loan model trained successfully'})

@app.route('/loan_predict', methods=['POST'])
def predict_loan():
    data = request.get_json()
    features = np.array([[data['age'], data['balance']]])
    prediction = loan_model.predict(features)[0]
    return jsonify({'prediction': int(prediction)})



if __name__ == '__main__':
    app.run(debug=True)
