# routes.py
from flask import render_template, request
from App.main import app
from App.scraper import scrape_url
from App.predictor import predict_text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form.get("url")
        result = scrape_url(url)

        if result["success"]:
            prediction = predict_text(result["text"])
        else:
            prediction = None

        return render_template("index.html", result=result, prediction=prediction)
    return render_template("index.html", result=None, prediction=None)