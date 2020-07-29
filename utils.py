from imports import *


def fetch_img_lbl(label_df,id_,path_data):
  """
  Inputs: df and id.
  Outputs: PIL image and its center location
  """
  imgname = label_df.imgName
  if imgname[id_][0] == 'A':
    folder = 'AMD'
  else:
    folder = 'Non-AMD'
  
  img = Image.open(f'{path_data}/Training400/{folder}/{imgname[id_]}')

  x = label_df['Fovea_X'][id_]  
  y = label_df['Fovea_Y'][id_]
  centers = (x,y)
  return img,centers

def plot_img(img,centers,w_h=(50,50),thickness=2):
  w,h = w_h
  x,y = centers

  draw = ImageDraw.Draw(img)
  draw.rectangle(((x-w/2,y-h/2),(x+w/2,y+h/2)),outline='blue',width = thickness)
  plt.imshow(np.asarray(img))

def show_img(img,label=None):
  img_np = img.numpy().transpose((1,2,0))
  plt.imshow(img_np)
  if label is not None:
    label = label_rescaler(label,img.shape[1:])
    x,y = label
    plt.plot(x,y,'r+',markersize=15)

def resize_img(img,label = (0.,0.),target_size = (256,256)):
  w,h = img.size
  w_targ,h_targ = target_size
  x,y = label
  img_new = TF.resize(img,target_size)
  label_new = (x/w)*w_targ , (y/h)*h_targ
  return img_new, label_new

def random_hflip(img,label):
  w,h = img.size
  x,y = label

  img = TF.hflip(img)
  label = w-x, y
  return img,label

def random_vflip(img,label):
  w,h = img.size
  x,y = label

  img = TF.vflip(img)
  label = x, h-y
  return img,label

def random_shift(img,label,max_translate = (0.15,0.15)):
  w,h = img.size
  max_t_w,max_t_h = max_translate
  x,y = label
  trans_coef = np.random.rand()*2 - 1
  w_t, h_t = int(trans_coef*max_t_w*w),int(trans_coef*max_t_h*w)

  img = TF.affine(img,translate=(w_t,h_t),scale=1,shear=0,angle=0)
  label = x + w_t, y + h_t
  return img,label

def label_scaler(a,b):
  lbl_scaled = [ai/bi for ai,bi in zip(a,b)]
  return lbl_scaled

def label_rescaler(a,b):
  lbl_rescaled = [ai*bi for ai,bi in zip(a,b)]
  return lbl_rescaled
