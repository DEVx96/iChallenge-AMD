from utils import *
from transforms import *

class AMD_classification_dataset(Dataset):
  
  def __init__(self,path_data,transforms,tf_params):
    df = pd.read_excel(f'{path_data}/Training400/Fovea_location.xlsx',index_col="ID")
    df['labels'] = 0* df.shape[0]
    self.imgName = df["imgName"]
    self.path = [0]*len(self.imgName)
    
    for id in df.index:
      if self.imgName[id][0] == "A":
        folder = "AMD"
        df.iloc[id-1,-1] = 1
      else:
        folder = "Non-AMD"
        df.iloc[id-1,-1] = 0
      self.path[id-1] = f'{path_data}/Training400/{folder}/{self.imgName[id]}'
    
    self.labels = df["labels"].values
    self.transform = transforms
    self.tf_params = tf_params 
  
  def __len__(self):
    return len(self.imgName)
  
  def __getitem__(self,idx):
    img = Image.open(self.path[idx])
    label = self.labels[idx]
    img = self.transform(img,self.tf_params)
    return img,label

class AMD_dataset(Dataset):
  
  def __init__(self,path_data,transforms,tf_params):
    df = pd.read_excel(f'{path_data}/Training400/Fovea_location.xlsx',index_col="ID")
    ids = df[df["Fovea_Y"] == 0 ].index
    df.drop(ids,inplace=True)
    new_ids = np.arange(0,395)
    df.set_index(new_ids,inplace = True)

    self.labels = df[["Fovea_X","Fovea_Y"]].values
    self.imgName = df["imgName"]
    self.path = [0]*len(self.imgName)
    for id in df.index:
      if self.imgName[id][0] == "A":
        folder = "AMD"
      else:
        folder = "Non-AMD"
      self.path[id-1] = f'{path_data}/Training400/{folder}/{self.imgName[id]}'
    
    self.transform = transformer
    self.tf_params = tf_params 
  
  def __len__(self):
    return len(self.imgName)
  
  def __getitem__(self,idx):
    img = Image.open(self.path[idx])
    label = self.labels[idx]
    img,label = self.transform(img,label,self.tf_params)
    return img,label
