from imports import *

class Conv_layer(nn.Module):
  def __init__(self,ni,nf,pool = True,stride=2):
    super(Conv_layer,self).__init__()
    self.conv = nn.Conv2d(ni,nf,kernel_size=3,stride=stride,padding=1)
    self.bn = nn.BatchNorm2d(nf)
    self.pool = pool
  def forward(self,x):
    x = self.conv(x)
    x = self.bn(x)
    x = F.relu(x)
    if self.pool:
      x = F.avg_pool2d(x,2,2)
    return x

class Resblock(nn.Module):
  def __init__(self,n):
    super(Resblock,self).__init__()
    self.conv1 = Conv_layer(n,n,pool = False,stride=1)
    self.conv2 = Conv_layer(n,n,pool = False,stride=1)
  
  def forward(self,x):
    return x + self.conv2(self.conv1(x))

class Model(nn.Module):
  def __init__(self,input_channel,input_filters):
    super(Model,self).__init__()

    ni = input_channel
    nf = input_filters
    self.conv1 = Conv_layer(ni,nf,pool=False)
    self.res1 = Resblock(nf)
    self.conv2 = Conv_layer(nf,2*nf)
    self.res2 = Resblock(2*nf) 
    self.conv3 = Conv_layer(2*nf,4*nf)
    self.res3 = Resblock(4*nf)
    self.conv4 = Conv_layer(4*nf,8*nf)
    self.res4 = Resblock(8*nf)
    self.fc1 = nn.Linear(8*nf,2)

  def forward(self,x):

    x = self.res1(self.conv1(x))
    x =  self.res2(self.conv2(x))
    x =  self.res3(self.conv3(x))
    x =  self.res4(self.conv4(x))
    x = F.avg_pool2d(x,2,2)
    x =  x.reshape(x.size(0), -1)
    x = self.fc1(x)
    return x

