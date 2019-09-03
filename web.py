import torch
import torchvision.utils
import torch.nn as nn
import torch.nn.functional as F
import flask
import numpy as np
import json
import os

conf = {}
all_min = []
all_max = []
all_range = []
initial = []
slider_range_min = 1
slider_range_max = 20
bin_size = []
app = flask.Flask(__name__)

def to_img(x, channels, n):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), channels, n, n)
    return x

def sliders_to_vals(slider_vals):
  vals = np.array(slider_vals, dtype=float)
  vals = vals - slider_range_min
  vals = all_min + vals*bin_size
  return vals

def vals_to_sliders(vals):
  return (np.rint((vals-all_min) / bin_size)+slider_range_min)[0].astype(int).tolist()

def img_from_vals(vals):
  r = np.array(vals, dtype=float)
  r.shape = (1, 1, -1)
  r = torch.from_numpy(r).float()
  img = net.decoder(r)
  img = to_img(img.data, net.num_channels, net.img_size)
  return img

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
    img = img_from_vals(initial)
    torchvision.utils.save_image(img, 'static/img.png')
    path = 'img.png'
    return flask.render_template('index.html', code_size=code_size, slider_range_min=slider_range_min,slider_range_max=slider_range_max, num_cols=num_cols, num_rows=num_rows, slider_initial=slider_initial)

@app.route('/get_image_path')
def get_image_path():
  # Read vals from the front end
  slider_vals = flask.request.args.getlist('slider_vals[]')
  # Map all values onto their valid range, respectively
  vals = sliders_to_vals(slider_vals)
  img = img_from_vals(vals)
  torchvision.utils.save_image(img, 'static/img.png')
  path = 'img.png'
  # Return the path to the image we just created
  return flask.jsonify(result=path)

# Will return None if there are no datasets or if the user choses to exit
def pick_state():
  static_folder = 'static'
  states = []
  # Get the name from the state (.pth) files
  set_names = [f[:f.find('_')] for f in os.listdir(static_folder) if f.endswith('.pth') and f.find('_')>-1]
  # Add states. State files (.pth) and code input files (.npy) should exist in pairs
  for set_name in set_names:
    fstate = os.path.join(static_folder, set_name+'_state.pth')
    fcode = os.path.join(static_folder, set_name+'_code.npy')
    if os.path.exists(fstate) and os.path.exists(fcode):
      states.append((set_name, fstate, fcode))
  if len(states)==0:
    return None
  while True:
    print(f'0: Exit')
    for idx, state in enumerate(states):
      print(f'{1 + idx}: {state[0]}')
    selected = input("Enter your dataset of choice: ")
    if selected.isdigit():
      selected = int(selected)
      if selected==0:
        return None
      elif selected>=1 and selected<=len(states):
        return states[selected-1]
  
def set_vals_range(code_input):
  """ Our sliders have a discrete number of possible values. There's a mapping
  between slider values and actual values used as input fir the latent space.
  This could be done by mapping the values to slider values so that the min/max
  slider value correspond to min/max input values. The issue with this, given we
  use a linear scaling on the sliders, is that the initial input values is 
  unlikely to be achive with the sliders. The solution is to modify min/max values
  so that the initial value is on the sliders. Steps are:
  1. bin_size = (max-min)/num_bins
  2. n = index of the bin that contains our initial value
  3.  min = initial - n*bin_size
      max = initial + (num_bins-n)*bin_size
  """
  global initial, all_min, all_max, all_range, bin_size
  initial = code_input[np.random.randint(0, code_input.shape[0])][0]
  all_min = np.min(code_input, axis=0)
  all_max = np.max(code_input, axis=0)
  all_range = all_max - all_min
  num_bins = slider_range_max-slider_range_min+1
  bin_size = all_range / num_bins
  n = np.rint((initial-all_min) / bin_size)
  all_min = initial - n*bin_size
  all_max = initial + (num_bins-n)*bin_size

if __name__=="__main__":
  state = pick_state()
  if state!=None:
    set_name, fstate, fcode = state
    with open('conf.json') as json_file:
      conf = json.load(json_file)
    code_input = np.load(fcode)
    set_vals_range(code_input)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = torch.load(fstate, map_location=torch.device(device))
    # Make Flask server reachable from local network
    app.run(host='0.0.0.0')
