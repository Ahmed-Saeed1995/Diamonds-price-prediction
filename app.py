import joblib
from flask import Flask, render_template, url_for, request

app = Flask(__name__)
model = joblib.load('Diamond predictions with decision tree.h5')


@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html")

@app.route("/predict", methods=["GET"])
def predict ():
    list_data = [
                request.args.get("carat"),
                
                request.args.get("x"),
                request.args.get("y"),
                request.args.get("z"),

                request.args.get("cut_grade"),
                request.args.get("color"),
                request.args.get('clarity_quality')]

    numeric_data = [float(val) for val in list_data]
    diamond_price = model.predict([numeric_data])[0]
    return render_template("index.html", diamond_price = (diamond_price).round(2))





if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port='8000')

