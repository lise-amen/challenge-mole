import os
from flask import request, Flask, flash, redirect, url_for, render_template
from datetime import datetime
from werkzeug.utils import secure_filename


app = Flask(__name__, template_folder="templates")

app.config['IMAGE_UPLOADS'] = "/home/melvin/Documents/BeCode/Projects/challenge-mole/app/static/image/uploads"
app.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPG", "JPEG", "PNG", "BMP"]

@app.route('/')
def home():
	return render_template("start.html")

def allowed_image(filename):

	if not "." in filename:
		return False
	
	ext = filename.rsplit(".", 1)[1]

	if ext.upper() in app.config["ALLOWED_IMAGE_EXTENSIONS"]:
		return True
	
	else:
		return False

@app.route('/upload', methods=["GET", "POST"])
def upload_image():

	if request.method == "POST":

		if request.files:

			image = request.files["image"]

			if image.filename == "":
				print("Image must have a filename")
				return redirect(request.url)
			
			if not allowed_image(image.filename):
				print("That image extension is not allowed")
				return redirect(request.url)
			
			else:
				filename = secure_filename(image.filename)

				image.save(os.path.join(app.config["IMAGE_UPLOADS"], filename))

			print("Image saved")

			return redirect("/uploaded")
	return render_template('upload.html')

@app.route("/uploaded", methods=["GET", "POST"])
def uploaded_image():
	return "Ok tu as upload une image"

@app.route("/authors")
def authors():
	return render_template("authors.html")

if __name__ == '__main__':
	app.run(debug=True, port=5000)
