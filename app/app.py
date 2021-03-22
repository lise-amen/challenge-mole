import os
from flask import request, Flask, flash, redirect, url_for, render_template
from datetime import datetime
from werkzeug.utils import secure_filename


app = Flask(__name__, template_folder="templates")
UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'BMP'])

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
	return render_template("base.html")

@app.route('/upload')
def upload_file():
   return render_template('upload.html')
	
@app.route('/uploader', methods = ['GET', 'POST'])
def upload_files():
   if request.method == 'POST':
      f = request.files['file']
      f.save(secure_filename(f.filename))
      return 'file uploaded successfully'

if __name__ == '__main__':
	app.run(port=5000)
	app.run(debug=True)
