import pandas as pd
import numpy as np
import os, sys
# import seaborn as sns
import matplotlib.pyplot as plt
import glob
# from PIL import Image, ImageOps
# from plotly.subplots import make_subplots
# from warnings import filterwarnings
from io import StringIO, BytesIO
import streamlit as st
import torch
import cv2
# import argparse
from tempfile import NamedTemporaryFile
import warnings

warnings.filterwarnings('ignore')

# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--img", required=False, help="path to image")
# args = vars(ap.parse_args())

def load_image(img_path, resize=True):
  img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
  img = cv2.resize(img, (300, 300), interpolation = cv2.INTER_AREA)
  # img = Image.open(img_path)
  # img = ImageOps.exif_transpose(img.resize((300, 300), Image.ANTIALIAS))
  return img

def show_grid(image_paths):
  images = [load_image(img) for img in image_paths]
  images = torch.as_tensor(images)
  images = images.permute(0, 3, 1, 2)
  # grid_img = torchvision.utils.make_grid(images, nrow=11)
  plt.figure(figsize=(24, 12))
  # plt.imshow(grid_img.permute(1, 2, 0))
  plt.axis('off')

def format_predictions(img_path, results, num_classes=8):
  """
  Format the predictions as they
  are not in order of the meter
  reading
  For some reason inside this app, the 
  detect script generates more rows than
  it does when tested in Colab. So, I have
  taken the top num_classes records
  """
  if not isinstance(results, str):
    df = results.pandas().xyxy[0].head(num_classes)
  else:
    df = pd.read_csv(results, sep=' ', header=None)
  df.columns = ['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class', 'name']
  df.sort_values('xmin', ascending=True, inplace=True)
  img = load_image(img_path)

  fig = plt.figure(figsize=(7, 7))
  plt.imshow(img)
  reading = ''.join(str(s) for s in df['class'].values)
  if len(reading) > 5:
    reading = reading[:-3] + '.' + reading[-3:]
  try:
    reading = float(reading)
    plt.title('Reading ' +  str(reading) + " m\u00b3")
  except:
    plt.title('Is this a valid water meter image...?')
  plt.axis('off')
  st.pyplot(fig)

@st.cache
def load_model(src, path, device, reload=False):
  return torch.hub.load(src, 'custom', path=path, device=device, force_reload=reload)

yolo_path = os.path.join(os.getcwd(), 'yolov5')
# location = 'runs/train/yolov5x_water_meter/weights/best.pt'
location = 'weights/best.pt'
best_run = os.path.join(os.getcwd(), 'yolov5', location)
device = torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'cpu'

#model = torch.hub.load('ultralytics/yolov5', 'custom', path=location, device=device, force_reload=False)
src = 'ultralytics/yolov5'
model = load_model(src, path=location, device=device) 

buffer = st.file_uploader("Upload water meter reading image", type=['png', 'jpeg', 'jpg'])
@st.cache(ttl=24*3600, suppress_st_warning=True, show_spinner=False)
def predict(inp):
# results = model(args['img'])
# format_predictions(args['img'], results)
  if inp:
    # https://discuss.streamlit.io/t/image-upload-problem/4810/5
    temp_file = NamedTemporaryFile(delete=True)
    temp_file.write(inp.getvalue())
    results = model.to(device)(temp_file.name)
    format_predictions(temp_file.name, results)

predict(buffer)
