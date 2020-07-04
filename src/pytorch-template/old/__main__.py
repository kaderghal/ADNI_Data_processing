
import sys

import torch
from models.baseline_3D_single import SE_HIPP_3D_Net
from data_loader.data_loader import Dataset_ADNI_Folder
from data_loader.data_loader import pickle_loader


from torch import nn
from torch import optim
from torchsummary import summary

###############################################################################################################
# server
###############################################################################################################
sys.path.append('/data/ADERGHAL/code-source/ADNI_Data_processing/src/data_processing/')
root_path = '/data/ADERGHAL/ADNI_workspace/results/ADNI_des/F_28P_F10_MS2_MB10D/HIPP/3D/AD-NC/'



#==========================================================================
# Function: Main definition 
#========================================================================== 
def main():

    # parames for data
    id_device = 1
    params_num_workers = 4
    batch_size = 64
    num_classes = 2
    save_frequency = 2
    learning_rate = 0.00001
    num_epochs = 500
    weight_decay = 0.0001
    
    train_losses, test_losses = [], []
    running_loss = 0
    steps = 0
    print_every = 35 # 175/5
    
    # select device
    device = torch.device("cuda:" + str(id_device) if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    print("using device :", device)
    model = SE_HIPP_3D_Net().to(device)

    # weights initialization    
    # model.apply(weights_init)

    # DataFolder
    train_data = Dataset_ADNI_Folder(root=root_path + 'train/', loader=pickle_loader, extensions='.pkl', transform=None)
    valid_data = Dataset_ADNI_Folder(root=root_path + 'valid/', loader=pickle_loader, extensions='.pkl', transform=None)
    test_data  = Dataset_ADNI_Folder(root=root_path + 'test/' , loader=pickle_loader, extensions='.pkl', transform=None)
    
 
    # Dataloader   
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=params_num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True, num_workers=params_num_workers)
    test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=True, num_workers=params_num_workers)
    
   
    # net = LeNet()
    summary(model, (28, 28, 28))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
      
    #scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.1)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    valid_acc = []

    running_loss = 0.0
    for epoch in range(num_epochs):
        for i, (d1, d2, v, labels) in enumerate(train_loader):
            
            #
            steps += 1

            # # forward + backward + optimize
            # print("d1 size:", d1.size())
            # d1 = torch.unsqueeze(d1, 1).to(device, dtype=torch.float)
            d1 = d1.to(device, dtype=torch.float)
            # print("d1 size:", d1.size())
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()   

            outputs = model(d1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()


            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
            # acc_list.append((correct / total) * 100)
            
            
            if steps % print_every == 0:
                acc_list.append((correct / total) * 100)
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for i, (v_d1, v_d2, v_v, v_labels) in enumerate(valid_loader):
                        # v_d1 = torch.unsqueeze(v_d1, 1).to(device, dtype=torch.float)
                        v_d1 = v_d1.to(device, dtype=torch.float)
                        v_labels = v_labels.to(device)
                        v_outputs = model(v_d1)
                        batch_loss = criterion(v_outputs, v_labels)           
                        test_loss += batch_loss.item()                    
                        ps = torch.exp(v_outputs)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == v_labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            
            
                # train_losses.append(running_loss/len(train_loader))
                train_losses.append(running_loss/print_every)
                test_losses.append(test_loss/len(valid_loader))    
                
                                
                print(f"Epoch {epoch+1}/{num_epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Train accuracy: {(correct / total) * 100:.3f}.. "
                  f"Test loss: {test_loss/len(valid_loader):.3f}.. "
                  f"Test accuracy: {(accuracy/len(valid_loader) * 100):.3f}")
                
                valid_acc.append((accuracy/len(valid_loader) * 100))
                
                running_loss = 0
                model.train()
                
               # scheduler.step()


    
    plt.plot(acc_list, label='Training accu')
    plt.plot(valid_acc, label='Validation accu')
    
    plt.legend(frameon=False)
    plt.show()


    plt.plot(train_losses, label='Training loss')
    plt.plot(test_losses, label='Validation loss')
    plt.legend(frameon=False)
    plt.show()



    print('Finished Training')



#==========================================================================
# Start : __Main__
#==========================================================================    
if __name__ == '__main__':
    main()