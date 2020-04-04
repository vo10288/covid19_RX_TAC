import face_recognition
import subprocess
from flask import Flask, jsonify, request, redirect, render_template
import cv2
from datetime import datetime
import time
import os
from math import sqrt
from sklearn import neighbors
from os import listdir
from os.path import isdir, join, isfile, splitext
import pickle
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from face_recognition import face_locations
from face_recognition.face_recognition_cli import image_files_in_folder
#from fabric import run, env, sudo, put
import ftplib
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

###################################
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os
import hashlib

##################################
print("OLD DATASET")
os.getcwd()

# You can change this to any folder on your system
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__,  static_folder='/home/vision/Downloads/COVID_cnn-keras/', template_folder='/home/vision/Downloads/COVID_cnn-keras/')


PEOPLE_FOLDER = os.path.join('')##
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER ##

def allowed_file(filename):
	#cv2.imwrite('./uploads/'+filenam, filename)
	return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    # Check if a valid image file was uploaded
	if request.method == 'POST':
		if 'file' not in request.files:
			return redirect(request.url)
        
		file = request.files['file']
			
		
		if file.filename == '':
			return redirect(request.url)

		if file and allowed_file(file.filename):
            
            # The image file seems valid! Detect faces and return the result.
			return detect_faces_in_image(file)


#        if 'file2' not in request.files:
#            return redirect(request.url)
        
#        file = request.files['file2']
			
		
#        if file.filename == '':
#            return redirect(request.url)

#        if file and allowed_file(file.filename):
            
            # The image file seems valid! Detect faces and return the result.
#			return detect_faces_in_image2(file)
    # If no valid image file was uploaded, show the file upload form:
	return '''
    <!doctype html>
    <head>
	<title>
	Research the COVID-19 with RX IMAGE -- ATTENTION THIS METOD IS EXPERIMENTAL !! diagnoses can only be made by specialist doctors
	</title>
	</head>
	<table width="100%" bgcolor="darkblue">
	<tr><td>
	<h2>Research the COVID-19 with RX IMAGE -- ATTENTION THIS METOD IS EXPERIMENTAL !! diagnoses can only be made by specialist doctors</h2>
	
	<h3>Upload a photo and see the prediction of Artificial Intelligence !</h3>
	<h5>Created by Antonio Vision Broi -- Copyright Licenze M.I.T.</h5>
	</td></tr>
	</table>
    
    <body style="width:100%;height:100%;" bgcolor="white" text="ciano" background="ai.jpg"> 
    <table><tr><td>
    </td></tr></table><br><br>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file"  style="font-size:120%;width:40%;height:100%;bgcolor:yellow;color:black;" ><br><br>
      <input type="submit" value="Upload" style="font-size:120%;width:20%;height:100%;color:black;bgcolor:yellow;" >
    </form>
    
    '''
#<img src="http://www.sapere.it/mediaObject/icone-sapere/IMMAGINI-VARIE/pillole/batman0/original/batman.jpg" target="_blank" width="80%">
face_found = False
is_obama = False    

#def detect_faces_in_image2(file_stream):
#    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
#    known_face_encoding = [-0.06474961,  0.10491645, -0.01633071, -0.02837753, -0.05284042,
#        0.03039825,  0.02796371, -0.06017469,  0.23516014, -0.0030535 ,
#        0.24534011, -0.06778147, -0.26664701, -0.08831722,  0.04156463,
#        0.13598767, -0.130153  , -0.15152329, -0.00189391, -0.08122989,
#       -0.00969385, -0.00242687,  0.05609196,  0.04444347, -0.12514067,
#       -0.36383939, -0.06363531, -0.20127141,  0.05413001, -0.07592988,
#       -0.13241455,  0.07765745, -0.13463351, -0.14879228,  0.11749732,
#        0.13707586, -0.094152  , -0.03047094,  0.25445738,  0.01599296,
#       -0.10969914,  0.00723559,  0.04131952,  0.33481261,  0.11419167,
#        0.04213948,  0.05550952, -0.02144238,  0.03862203, -0.23548217,
#        0.13977587,  0.14830808,  0.13343999,  0.07722504,  0.08605643,
#       -0.22706951, -0.04940255,  0.13315007, -0.24466926,  0.14879599,
#        0.08725536, -0.04561751, -0.04798657, -0.05765189,  0.31570077,
#        0.1345862 , -0.1704136 , -0.09104119,  0.13871641, -0.07073835,
#       -0.0039288 ,  0.04798311, -0.16081105, -0.23755735, -0.21088155,
#        0.13249612,  0.36320627,  0.12433351, -0.16322078,  0.01439504,
#       -0.04432993, -0.04595296,  0.00706228,  0.09307377, -0.06007439,
#        0.06086485, -0.06979594,  0.0246887 ,  0.11906222, -0.0232532 ,
#       -0.05757546,  0.20526388, -0.0868334 ,  0.06030093,  0.03324999,
#        0.07646361, -0.07814351, -0.03306082, -0.17319438, -0.04055523,
#        0.03495843, -0.17359141, -0.00563912,  0.09193631, -0.21365114,
#        0.06294478, -0.02878716, -0.04152306, -0.08320279, -0.01282814,
#       -0.07075235,  0.00879254,  0.19283667, -0.35026604,  0.24284841,
#        0.20366861,  0.07143958,  0.17847557,  0.03893464, -0.03109722,
#        0.09138451, -0.0591217 , -0.18514556, -0.06454267,  0.02265352,
#        0.02509806,  0.04294147,  0.083833]
def change_image():		
		os.execl(sys.executable, sys.executable, *sys.argv)        


        
def detect_faces_in_image(file_stream):
    # Pre-calculated face encoding of Obama generated with face_recognition.face_encodings(img)
    
    #######################################
	if not os.path.exists('uploadRx'):
		os.mkdir('uploadRx')
		
	filenames = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")	
	
	# load the image
	image = face_recognition.load_image_file(file_stream)
	#image = cv2.imread(file_stream)
	#unknown_face_encodings = face_recognition.face_encodings(image)
	pil_image = Image.fromarray(image)
	
	if not os.path.exists('uploadRx/'+filenames):
		os.mkdir('uploadRx/'+filenames)
	pil_image.save('uploadRx/'+filenames+'/image.jpg')
	
	
	#readFile = openedFile.read()
	
	# hash image file uploading
	filess = 'uploadRx/'+filenames+'/image.jpg' # Location of the file (can be set a different way)
	BLOCK_SIZE = 65536 # The size of each read from the file

	file_hash = hashlib.sha256() # Create the hash object, can use something other than `.sha256()` if you wish
	with open(filess, 'rb') as f: # Open the file to read it's bytes
		fb = f.read(BLOCK_SIZE) # Read from the file. Take in the amount declared above
		while len(fb) > 0: # While there is still data being read from the file
			file_hash.update(fb) # Update the hash
			fb = f.read(BLOCK_SIZE) # Read the next block from the file

	print (file_hash.hexdigest()) # Get the hexadecimal digest of the hash
	
	command = ('python3 classifyCovid_2.py -i '+'uploadRx/'+filenames+'/image.jpg')
	subprocess.Popen(command, shell=True)
	time.sleep(15)
	
	
#######################################

########################################################################################################################################
	label = open("label.csv","r").read()
	print(label)
	full_filename1 = ('uploadRx/'+filenames+'/image.jpg')#'uploadRx/diagnosed/diagnosed2.jpg'
	sha256 = (file_hash.hexdigest())
	timestamp = str(filenames)
#	full_filename2 = str(directori+'/'+filenam)
#	full_filename3 = str(name)      
#	print(full_filename1)
	return  render_template("return6_ftp.html", user_image1 = full_filename1, user_image3 = label, user_image4 = sha256, user_image5 = timestamp) #, user_image2 = full_filename2, user_image3 = full_filename3)
	change_image()	
	
#	return  '''
#    <!doctype html>
#    <body style="width:100%;height:100%;" bgcolor="black" text="yellow" background="http://www.broi.it/broi/terra.jpg" position="center"> 
#    <title>Is this a picture of your TARGET ?</title>
#    <h3>THANK YOU FOR your Upload a picture and see THE VERIFY & IDENTIFICATION !</h3>
#    <table><tr><td>
#    <a href="http://www.broi.it/ew/match.jpg"><img src="http://www.broi.it/ew/match.jpg"  height="400px"></a>
#    </td></tr></table>
  
#    <form method="POST" enctype="multipart/form-data">
#      <input type="file" name="file"  style="font-size:120%;width:40%;height:100%;bgcolor:yellow;color:black;" ><br><br>
#      <input type="submit" value="Upload" style="font-size:120%;width:20%;height:100%;color:black;bgcolor:yellow;" >
#    </form>
#    </body>
#    </html>
    
    
#   '''
#### <img src="http://www.sapere.it/mediaObject/icone-sapere/IMMAGINI-VARIE/pillole/batman0/original/batman.jpg" target="_blank" width="80%">
	 # Return the result as json

	
	result = {
        "face_found_in_image ?": face_found,
        "the_result_of_verification_is": name,
        "totale_results": loc[0],
        #"totale_results1": loc[1],
        #"totale_results2": loc[2],
        #"totale_results3": loc[3]
    }
	return jsonify(result) 



if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9999, debug=True)
