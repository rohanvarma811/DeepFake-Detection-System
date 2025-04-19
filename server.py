from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# ADDED
app = Flask(__name__)

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition

# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time

import sys

# 'nn' Help us in creating & training of neural network
from torch import nn

# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------------------------------------------------------- DONE IMPORTING LIBRARIES -------------------------

from dotenv import load_dotenv
import os
# Load environment variables
load_dotenv()

BASE_DIR = os.path.abspath(os.path.dirname(__file__))  # Root directory of the project

# Environment-based configuration
UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv("UPLOAD_FOLDER", "Uploaded_Files"))
MODEL_PATH = os.path.join(BASE_DIR, os.getenv("MODEL_PATH", "DeepFake_Detection/model/df_model.pt"))
SECRET_KEY = os.getenv("SECRET_KEY", "default_secret_key")

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = SECRET_KEY

# --------------------------------------------------------------------------------------------------------------------------- DONE ADDING NEW CODE SNIPPET -------------------------


UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []

app = Flask("__main__", template_folder="templates")
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creating Model Architecture

class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# For prediction of output  
# ORIGINAL CODE
'''
def predict(model, img, path='./'):
  # use this command for gpu    
  # fmap, logits = model(img.to('cuda'))
  fmap, logits = model(img.to())
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item()*100
  print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
  return [int(prediction.item()), confidence]
'''


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    for i, frame in enumerate(self.frame_extract(video_path)):
      faces = face_recognition.face_locations(frame)
      try:
        top,right,bottom,left = faces[0]
        frame = frame[top:bottom, left:right, :]
      except:
        pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path)
    success = 1
    while success:
      success, image = vidObj.read()
      if success:
        yield image


def predict(model, img, path='./'):
    # use this command for gpu    
    # fmap, logits = model(img.to('cuda'))
    fmap, logits = model(img.to())
    params = list(model.parameters())
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    
    # Get the predicted class and confidence score
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    print('confidence of prediction: ', confidence)
    
    # Return the prediction and confidence
    return [int(prediction.item()), confidence]


def detectFakeVideo(videoPath):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    path_to_videos = [videoPath]

    # Initialize the dataset
    video_dataset = validation_dataset(path_to_videos, sequence_length=20, transform=train_transforms)
    
    # Load the model
    model = Model(2)
    path_to_model = './model/df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()

    # Get the prediction for the video
    prediction = predict(model, video_dataset[0], './')

    # Check the confidence and classify the video
    if prediction[1] < 98:
        output = "REAL"
    else:
        output = "FAKE"
    
    # Return the classification and confidence
    return [output, prediction[1]]


@app.route('/', methods=['POST', 'GET'])
def homepage():
  if request.method == 'GET':
     return render_template('./index.html')
  return render_template('./index.html')


@app.route('/Detect', methods=['POST', 'GET'])
def DetectPage():
    if request.method == 'GET':
        return render_template('./index.html')
    
    if request.method == 'POST':
        # Get the video file from the form
        video = request.files['video']
        print(video.filename)
        
        # Secure the filename and save the video
        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        
        # Get the path to the uploaded video
        video_path = "./Uploaded_Files/" + video_filename
        
        # Get prediction and confidence for the video
        prediction = detectFakeVideo(video_path)
        
        # Debug: Check the prediction output
        print(prediction)
        
        # Prepare the output based on the classification
        output = prediction[0]
        confidence = prediction[1]
        
        # Prepare data to be passed to the HTML template
        data = {'output': output, 'confidence': confidence}
        data = json.dumps(data)
        
        # Clean up by removing the video file after processing
        os.remove(video_path)
        
        # Render the template with the result
        # Here it returns the 2 output parameters to be rendered on main page
        return render_template('./index.html', data=data)
        
if __name__ == "__main__":
  # Get the PORT from the environment variable (or default to 3000)
  port = int(os.environ.get("PORT", 3000))
  app.run(host="0.0.0.0", port=port, debug=False)