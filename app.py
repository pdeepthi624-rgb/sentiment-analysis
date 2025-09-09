from flask import Flask, request, render_template
import pickle
import mysql.connector

# Create Flask app
app = Flask(__name__)

# Load your trained model and vectorizer
model_path = "models/sentiment_model.pkl"
vectorizer_path = "models/vectorizer.pkl"

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

# Database connection
db = mysql.connector.connect(
    host="localhost",
    user="root",         # default user in XAMPP
    password="",         # keep blank if no password
    database="sentiment_db"
)
cursor = db.cursor()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_text = request.form["text"]

        if not input_text.strip():
            return render_template("index.html", error="⚠️ Please enter some text!")

        # Transform input text
        input_vec = vectorizer.transform([input_text])

        # Predict sentiment
        prediction = model.predict(input_vec)[0]

        # Predict probability (confidence)
        if hasattr(model, "predict_proba"):
            proba = float(model.predict_proba(input_vec).max())
        else:
            proba = None

        # ✅ Save result into MySQL
        sql = "INSERT INTO sentiment_results (input_text, predicted_label, confidence_score) VALUES (%s, %s, %s)"
        values = (input_text, prediction, proba)
        cursor.execute(sql, values)
        db.commit()

        return render_template(
            "index.html",
            prediction=prediction,
            proba=proba,
            input_text=input_text,
        )

if __name__ == "__main__":
    app.run(debug=True)
