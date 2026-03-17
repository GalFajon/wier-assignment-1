from flask import Flask
import database.database as database
import os

from api.sites import bp as sites_bp
from api.pages import bp as pages_bp
from api.page_data import bp as page_data_bp
from api.images import bp as images_bp
from api.links import bp as links_bp

app = Flask(__name__)

app.register_blueprint(sites_bp)
app.register_blueprint(pages_bp)
app.register_blueprint(page_data_bp)
app.register_blueprint(images_bp)
app.register_blueprint(links_bp)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    try:
        database.init()
    except Exception:
        pass

    app.run(host="0.0.0.0", port=port)