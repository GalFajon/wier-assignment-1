from flask import Flask
from flask_httpauth import HTTPBasicAuth

import database.database as database
import os

from api.sites import bp as sites_bp
from api.pages import bp as pages_bp
from api.page_data import bp as page_data_bp
from api.images import bp as images_bp
from api.links import bp as links_bp

app = Flask(__name__)

# Auth initialization:
basic_auth = HTTPBasicAuth()

@basic_auth.verify_password
def verify_password(username, password):
    if username == os.environ.get("API_USER", "crawler") and password == os.environ.get("API_PASSWORD", "supersecret"):
        return "OK."
    return None

@basic_auth.error_handler
def basic_auth_error(status):
    return "Invalid credentials.", status

@app.before_request
@basic_auth.login_required
def require_auth_for_all_requests():
    return None

app.register_blueprint(sites_bp)
app.register_blueprint(pages_bp)
app.register_blueprint(page_data_bp)
app.register_blueprint(images_bp)
app.register_blueprint(links_bp)

# ADD PROPER FLASK RACE CONDITION HANDLING!!!

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    try:
        database.init()
    except Exception:
        pass

    app.run(host="0.0.0.0", port=port, debug=False, ssl_context='adhoc')