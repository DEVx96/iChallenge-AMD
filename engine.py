from imports import *

def train_parameters(num_epochs,optimizer,loss_func,train_dl,val_dl,lr_scheduler, path_models,
      scheduler_type = 0, sanity_check=True):
  params={
    "num_epochs": num_epochs,
    "optimizer": optimizer,
    "loss_func": loss_func,
    "train_dl": train_dl,
    "val_dl": val_dl,
    "sanity_check": sanity_check,
    "lr_scheduler": lr_scheduler,
    "scheduler_type": 0,
    "path_models":path_models,
  }
  return params


def cxcy2bbox(cxcy,w= 50./256,h = 50./256):
  w_tensor = torch.ones(cxcy.shape[0],1,device=cxcy.device) * w
  h_tensor = torch.ones(cxcy.shape[0],1,device=cxcy.device) * h
  cx = cxcy[:,0].unsqueeze(1)
  cy = cxcy[:,1].unsqueeze(1)
  boxes = torch.cat((cx,cy,w_tensor,h_tensor),-1) # x,y,w,h
  return torch.cat((boxes[:,:2] - boxes [:,2:]/2, boxes[:,:2] + boxes [:,2:]/2),1) # (xmin,ymin,xmax,ymax)

def metrics_batch(output,target):
  output = cxcy2bbox(output)
  target = cxcy2bbox(target)
  iou = torchvision.ops.box_iou(output,target)
  return torch.diagonal(iou,0).sum().item()

def loss_batch(loss_func, output, target, optim=None):
  loss = loss_func(output, target)
  with torch.no_grad():
    metric_b = metrics_batch(output,target)
    if optim is not None:
      optim.zero_grad()
      loss.backward()
      optim.step()
    return loss.item(), metric_b

def loss_epoch(model,loss_func,dataset_dl,sanity_check=False,optim=None):
    running_loss=0.0
    running_metric=0.0
    len_data=len(dataset_dl.dataset)

    for xb, yb in dataset_dl:
        yb=torch.stack(yb,1)
        yb=yb.type(torch.float32).to(device)
        
        
        output=model(xb.to(device))
        
       
        loss_b,metric_b=loss_batch(loss_func, output, yb, optim)
        
        running_loss+=loss_b
        
        if metric_b is not None:
          running_metric+=metric_b
        if sanity_check is True:
          break

    loss=running_loss/float(len_data)
    
    metric=running_metric/float(len_data)
    
    return loss, metric

def train_loop(model, params):

    num_epochs=params["num_epochs"]
    loss_func=params["loss_func"]
    optim=params["optimizer"]
    train_dl=params["train_dl"]
    val_dl=params["val_dl"]
    sanity_check=params["sanity_check"]
    lr_scheduler=params["lr_scheduler"]
    scheduler_type = params["scheduler_type"] # 0 for plateau, 1 for cosine
    best_model_path = f'{params["path_models"]}/best_model.pth'
    loss_history={
        "train": [],
        "val": [],
    }
    
    metric_history={
        "train": [],
        "val": [],
    }    
    
    
    torch.save(model.state_dict(),best_model_path)
    
    best_loss = float('inf')    
    
    for epoch in range(num_epochs):
        
        print('Epoch {}/{}'.format(epoch + 1, num_epochs ))   

        model.train()
        train_loss, train_metric=loss_epoch(model,loss_func,train_dl,sanity_check,optim)

        loss_history["train"].append(train_loss)
        metric_history["train"].append(train_metric)
        
        model.eval()
        with torch.no_grad():
            val_loss, val_metric=loss_epoch(model,loss_func,val_dl,sanity_check)
       
        loss_history["val"].append(val_loss)
        metric_history["val"].append(val_metric)   
        
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(),best_model_path)
                                   
        if scheduler_type == 0:
          lr_scheduler.step(val_loss)
        else:
          lr_scheduler.step(v)

                    
        print("train loss: %.6f, accuracy: %.2f" %(train_loss,100*train_metric))
        print("val loss: %.6f, accuracy: %.2f" %(val_loss,100*val_metric))
        print("-"*15) 
        

    
    model.load_state_dict(torch.load(best_model_path))
    
    return model, loss_history, metric_history
