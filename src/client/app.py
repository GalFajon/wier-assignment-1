from flask import Flask
import os
import utils.api_client as api_client # type: ignore

app = Flask(__name__)

@app.route('/')
def hello():
    return api_client.APIClient

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
