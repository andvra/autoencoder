import torch
import torchvision.utils
import torch.nn as nn
import torch.nn.functional as F
import flask
import numpy as np
import json

conf = {}
all_min = []
all_max = []
all_range = []
initial = []
slider_range_min = 1
slider_range_max = 20

app = flask.Flask(__name__)

def to_img(x, channels, n):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), channels, n, n)
    return x

def sliders_to_vals(slider_vals):
  vals = np.array(slider_vals, dtype=float)
  vals = vals - slider_range_min
  vals = all_min + np.multiply(all_range, vals)/(slider_range_max-slider_range_min+1)
  return vals

def vals_to_sliders(vals):
  block_size = all_range / (slider_range_max-slider_range_min+1)
  return ((vals-all_min) // block_size)[0].astype(int).tolist()

@app.route('/')
def index():
    # The client will be connected as soon as we return this page
    code_size = conf.get('code_size', 10)
    max_cols = 3
    split_size = 20
    num_cols = np.minimum(code_size//split_size+1, max_cols)
    num_rows = []
    for idx in range(num_cols-1):
      rows_in_col = code_size//num_cols+1
      num_rows.append(rows_in_col)
    num_rows.append(code_size-np.sum(num_rows))
    # Map network values to slider values
    slider_initial = vals_to_sliders(initial)
    return flask.render_template('index.html', code_size=code_size, slider_range_min=slider_range_min,slider_range_max=slider_range_max, num_cols=num_cols, num_rows=num_rows, slider_initial=slider_initial)

@app.route('/get_image_path')
def get_image_path():
  # Read vals from the front end
  slider_vals = flask.request.args.getlist('slider_vals[]')
  # Map all values onto their valid range, respectively
  vals = sliders_to_vals(slider_vals)
  r = np.array(vals, dtype=float)
  r.shape = (1, 1, -1)
  r = torch.from_numpy(r).float()
  img = net.decoder(r)
  img = to_img(img.data, net.num_channels, net.img_size)
  torchvision.utils.save_image(img, 'static/img.png')
  path = 'img.png'
  # Return the path to the image we just created
  return flask.jsonify(result=path)

if __name__=="__main__":
  with open('conf.json') as json_file:
    conf = json.load(json_file)
  code_input = np.load('static/code_input.npy')
  initial = code_input[np.random.randint(0, code_input.shape[0])][0]
  all_min = np.min(code_input, axis=0)
  all_max = np.max(code_input, axis=0)
  all_range = all_max - all_min
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  net = torch.load('static/state.pth', map_location=torch.device(device))
  app.run()