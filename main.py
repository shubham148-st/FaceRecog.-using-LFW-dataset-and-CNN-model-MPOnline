import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# dataset wrapper
class MyFaces(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        # convert to float
        img = self.x[idx].astype(np.float32)
        
        # add channel dim for grayscale
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)
            
        img = torch.tensor(img).permute(2, 0, 1)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return img, label

# basic cnn
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully connected layers
        self.fc1 = nn.Linear(128 * 6 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        
        # adaptive pool to make size fixed
        x = nn.functional.adaptive_avg_pool2d(x, (6, 4))
        x = x.view(-1, 128 * 6 * 4) 
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    # fetch data
    print("loading data...")
    lfw = fetch_lfw_people(min_faces_per_person=20, resize=0.4)
    x_data = lfw.images
    y_data = lfw.target
    classes = lfw.target_names.shape[0]

    # train test split
    xtrain, xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=0.2)

    # setup loaders
    train_ds = MyFaces(xtrain, ytrain)
    test_ds = MyFaces(xtest, ytest)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # model setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = CNN(classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30
    
    # store history for plotting
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    print("start training")
    for e in range(epochs):
        model.train()
        t_loss = 0
        correct = 0
        total = 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            opt.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, lbls)
            loss.backward()
            opt.step()
            
            t_loss += loss.item()
            _, pred = torch.max(out, 1)
            correct += (pred == lbls).sum().item()
            total += lbls.size(0)
            
        # evaluate on test set
        model.eval()
        v_loss = 0
        v_correct = 0
        v_total = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                loss = loss_fn(out, lbls)
                v_loss += loss.item()
                _, pred = torch.max(out, 1)
                v_correct += (pred == lbls).sum().item()
                v_total += lbls.size(0)
                
        t_acc = 100 * correct / total
        v_acc = 100 * v_correct / v_total
        
        history['train_loss'].append(t_loss/len(train_loader))
        history['val_loss'].append(v_loss/len(test_loader))
        history['train_acc'].append(t_acc)
        history['val_acc'].append(v_acc)
        
        print(f"Epoch {e+1} | Loss: {t_loss/len(train_loader):.3f} | Acc: {t_acc:.1f}% | Val Loss: {v_loss/len(test_loader):.3f} | Val Acc: {v_acc:.1f}%")

    # save weights
    torch.save(model.state_dict(), 'model.pth')
    
    # plot the results
    fig, ax1 = plt.subplots()
    ax1.plot(history['train_loss'], 'r-', label='Train Loss')
    ax1.plot(history['val_loss'], 'orange', label='Val Loss')
    ax1.set_ylabel('Loss')
    
    ax2 = ax1.twinx()
    ax2.plot(history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(history['val_acc'], 'c-', label='Val Acc')
    ax2.set_ylabel('Accuracy')
    
    fig.legend(loc='upper right')
    plt.title('Training Results')
    plt.savefig('result_graph.png')
    # plt.show() # comment out for now
