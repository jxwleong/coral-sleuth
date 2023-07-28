from flask import Flask, render_template
app = Flask(__name__, template_folder="ui/templates", static_folder="ui/static")

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
