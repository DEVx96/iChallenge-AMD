from utils import *

def transform_parameters(target_size = (256, 256), hflip = 0.5 ,vflip = 0.5,shift = 0.5,
        max_translate = (0.15, 0.15),brightness = 0.5, brightness_factor = 0.2,contrast = 0.5,
        contrast_factor = 0.2,gamma = 0.5,gamma_factor = 0.2,scale_label = True,test = False):
    
    """

    """
    if test:

        transform_params={
        "target_size" : target_size,
        "hflip" : 0.0,
        "vflip" : 0.0,
        "shift" : 0.0,
        "max_translate": max_translate,
        "brightness": 0.0,
        "brightness_factor": 0.0,
        "contrast": 0.0,
        "contrast_factor": 0.0,
        "gamma": 0.0,
        "gamma_factor": 0.0,
        "scale_label": scale_label,
        }

        return transform_params
    
    else:

        transform_params=  {
        "target_size" : target_size,
        "hflip" : hflip,
        "vflip" : vflip,
        "shift" : shift,
        "max_translate": max_translate,
        "brightness": brightness,
        "brightness_factor": brightness_factor,
        "contrast": contrast,
        "contrast_factor": contrast_factor,
        "gamma": gamma,
        "gamma_factor": gamma_factor,
        "scale_label": scale_label,
        }
        return transform_params

def transformer(img,label,params):
  img,label = resize_img(img,label,params['target_size'])
  if random.random() < params["hflip"]:
    img,label = random_hflip(img,label)
  if random.random() < params["vflip"]:
    img,label = random_vflip(img,label)
  if random.random() < params["shift"]:
    img,label = random_shift(img,label,params['max_translate'])
  
  if random.random() < params["brightness"]:
    brightness_factor=1+(np.random.rand()*2-1)*params["brightness_factor"]
    img = TF.adjust_brightness(img,brightness_factor)

  if random.random() < params["contrast"]:
    contrast_factor=1+(np.random.rand()*2-1)*params["contrast_factor"]
    img = TF.adjust_contrast(img,contrast_factor)

  if random.random() < params["gamma"]:
    gamma=1+(np.random.rand()*2-1)*params["gamma_factor"]
    img = TF.adjust_gamma(img,gamma)

  if params["scale_label"]:
    label=label_scaler(label,params["target_size"])
  
  img = TF.to_tensor(img)
  return img,label

def transformer_classifier(img,params):
  img = TF.resize(img,params['target_size'])

  if random.random() < params["hflip"]:
    img = TF.hflip(img)

  if random.random() < params["vflip"]:
    img = TF.vflip(img)

  if random.random() < params["brightness"]:
    brightness_factor=1+(np.random.rand()*2-1)*params["brightness_factor"]
    img = TF.adjust_brightness(img,brightness_factor)

  if random.random() < params["contrast"]:
    contrast_factor=1+(np.random.rand()*2-1)*params["contrast_factor"]
    img =TF.adjust_contrast(img,contrast_factor)

  if random.random() < params["gamma"]:
    gamma=1+(np.random.rand()*2-1)*params["gamma_factor"]
    img =TF.adjust_gamma(img,gamma)

  img = TF.to_tensor(img)
  return img