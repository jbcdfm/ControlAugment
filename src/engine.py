import torch
from torch.utils.data import DataLoader
import torch.nn as nn


# Optimizer update Function
def _optimize(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return

# Training Function for one epoch
def train_model(loader, optimizer, model, criterion, device):
    trn_corr = 0
    trn_loss = 0
    model.train()
    for b, (X_train, y_train) in enumerate(loader):
        X_train, y_train = X_train.to(device), y_train.to(device)
        y_pred = model(X_train)
        loss = criterion(y_pred, y_train)
        _, predicted = torch.max(y_pred.data, 1)
        trn_corr += (predicted == y_train).sum().item()
        trn_loss += loss.item()
        _optimize(optimizer, loss)
    return trn_corr, trn_loss

# Evaluate model on validation or test data
def test_model(loader, model, criterion, device):
    tst_corr = 0
    tst_loss = 0.0
    model.eval()
    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(loader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            y_pred = model(X_test)
            loss_test = criterion(y_pred, y_test)
            _, predicted = torch.max(y_pred.data, 1)
            tst_corr += (predicted == y_test).sum().item()
            tst_loss += loss_test.item()
            

    return tst_corr, tst_loss


# Evaluate model on test data (potential with TTA)
def test_model_tta(test_set, model, criterion, TTA_transforms, batch_sz, num_classes, device):
    
    test_size = len(test_set)
    y_out = torch.zeros([test_size*len(TTA_transforms),num_classes])
    y_true = torch.zeros([test_size,])
    
    for k in range(len(TTA_transforms)):
        test_set.transform = TTA_transforms[k]
        
        test_loader = DataLoader(
            test_set,
            batch_size = batch_sz, 
            shuffle=False,
            pin_memory=False,
            num_workers=0
            )
        
        # test_loader = DataLoader(test_set,batch_size = batch_sz,shuffle=False)
        model.eval()
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_loader):
                batch_length = len(y_test)
                X_test, y_test = X_test.to(device), y_test.to(device)
                out = nn.functional.softmax(model(X_test), dim=1)
                y_out[k*test_size+b*batch_sz:k*test_size+b*batch_sz+batch_length,:] = out#torch.argmax(out, dim=1)
                if k==0:
                    y_true[b*batch_sz:(b)*batch_sz+batch_length] = y_test
    
    y_pred = torch.argmax(y_out[:test_size], dim=1)              
    test_corr = (y_pred == y_true).sum().item()
    test_acc = test_corr/test_size*100
    
    y_TTA = y_out.view(len(TTA_transforms), test_size, num_classes).mean(dim=0)
    y_pred_TTA = torch.argmax(y_TTA[:test_size], dim=1)      
    test_corr = (y_pred_TTA == y_true).sum().item()
    test_acc_TTA = test_corr/test_size*100
    
                
    return test_acc, test_acc_TTA



# Eval Function for CtrlA dataset
def CtrlA_test_model(test_loader,model, criterion,device):
    model.eval()
    tst_corr = []
    with torch.no_grad():
        for b,(X_test,y_test) in enumerate(test_loader):
            X_test = X_test.to(device); y_test = y_test.to(device)
            y_pred = model.forward(X_test)
            _, predicted = torch.max(y_pred.data, 1)
            tst_corr.append((predicted == y_test).sum().item()) # Get result divided in batches

    return tst_corr