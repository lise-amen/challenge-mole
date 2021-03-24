import os
from flask import request, Flask, flash, redirect, url_for, render_template
from datetime import datetime
from werkzeug.utils import secure_filename

import torch
import torch.nn as nn

from PIL import Image

from app.Python_files.model_functions import mynet, data_transform


mole_detect = Flask(__name__, template_folder="templates")

mole_detect.config['IMAGE_UPLOADS'] = "app/static/image/uploads/"
mole_detect.config["ALLOWED_IMAGE_EXTENSIONS"] = ["JPG", "JPEG", "PNG", "BMP"]
mole_detect.config["MODEL_PATH"] = "app/static/model/state_dict_model.pt"

@mole_detect.route('/')
def home():
	return render_template("start.html")

def allowed_image(filename):

	if not "." in filename:
		return False
	
	ext = filename.rsplit(".", 1)[1]

	if ext.upper() in mole_detect.config["ALLOWED_IMAGE_EXTENSIONS"]:
		return True
	
	else:
		return False

@mole_detect.route('/upload', methods=["GET", "POST"])
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

				image.save(os.path.join(mole_detect.config["IMAGE_UPLOADS"], filename))

			print("Image saved")

			return redirect("/uploaded")
	return render_template('upload.html')

@mole_detect.route("/uploaded", methods=["GET", "POST"])
def uploaded_image():

	filename = os.listdir(mole_detect.config["IMAGE_UPLOADS"])

	img_path = mole_detect.config["IMAGE_UPLOADS"] + filename[0]

	image = data_transform(img_path)

	model = mynet()
	model.load_state_dict(torch.load(mole_detect.config["MODEL_PATH"]))
	model.eval()

	output = model(image)

	fsoftmax = nn.Softmax()

	proba = fsoftmax(output)

	acc, indice = torch.max(proba, dim=1)

	pred = str(indice.item())

	acc = str(round(acc.item() * 100, 2))

	os.remove(img_path)
	
	return render_template("uploaded.html", pred=pred, acc=acc)

@mole_detect.route("/authors")
def authors():
	return render_template("authors.html")

if __name__ == '__main__':
	mole_detect.run(debug=True, port=5000)
