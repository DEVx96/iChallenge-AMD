from imports import *

def amd_dataloader(train_ds,val_ds,bs_train = 16,bs_val = 16):

    split = ShuffleSplit(n_splits=1,test_size=0.2,random_state=10)
    ids = range(len(train_ds))
    for idx_train,idx_val in split.split(ids):
      print(len(idx_train),len(idx_val))

    train_ds = Subset(train_ds,idx_train)
    val_ds = Subset(val_ds,idx_val)
    
    train_dl = DataLoader(train_ds,batch_size=bs_train,shuffle=True)
    val_dl = DataLoader(val_ds,batch_size=bs_val,shuffle=True)

    return train_dl, val_dl



