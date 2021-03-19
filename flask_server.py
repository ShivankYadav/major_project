from flask import Flask, request
from flask_restful import Resource, Api 
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
api = Api(app)

class File(Resource):
    # Resource to handle file transfering.

    def post(self, name):
        # extracts the file from a POST request and saves it.
        f = request.files['file']
        filename = secure_filename(f.filename)
        
        save_path = os.path.join(os.getcwd(), filename)
        f.save(save_path)
        with open(save_path, 'r') as f_n:
            file_content = f_n.read()
        
        return file_content, 201
    
api.add_resource(File, '/file/<string:name>')

if __name__ == '__main__':
    app.run('0.0.0.0', port = 5000, debug =True)