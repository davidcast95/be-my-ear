from flask import Flask, render_template
from server.www.scripts.subproccess_thread import run_thread

app = Flask(__name__)
app.config.from_object(__name__)


app.config.update(dict(
    MODEL_DIR='model'
))
app.config.from_envvar('SETTINGS', silent=True)

@app.route("/")
def index():
    return render_template("home.html")

@app.route('/create_connection', methods=["GET"])
def create_connection():
    ip = '127.0.0.1'
    return render_template("create_connection.html",result=run_thread(['python','scripts/testing_conection.py']))


if __name__ == "__main__":
    app.run()
