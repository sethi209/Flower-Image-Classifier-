import torch
from torch import nn, optim
from torchvision import models
from utils import load_data, preprocess_image
import numpy
import json

def command_line_arguments(in_arg):
    if in_arg is None:
        print("* Doesn't Check the Command Line Arguments because 'get_input_args' hasn't been defined.")
    else:
        # prints command line agrs
        print("Command Line Arguments:\n     dir =", in_arg.data_dir, 
              "\n    save_dir =", in_arg.save_dir, "\n arch =", in_arg.arch,
              "\n    hidden_units =", in_arg.hidden_units, "\n learning_rate =", in_arg.learning_rate,
              "\n    epochs =", in_arg.epochs, "\n gpu =", in_arg.gpu)

def train_model(data_dir, save_dir, arch, epochs, learn_rate, hidden_units, use_gpu):
    trainloader, validloader, testloader, class_to_idx = load_data(data_dir)
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = 25088
    else:
        raise ValueError('Unsupported architecture: {}'.format(arch))
        
    for param in model.parameters():
        param.requires_grad = False

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)   
        
    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.3),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr = learn_rate)
    
    device = torch.device('cuda:0' if use_gpu == True else 'cpu')
    
    steps = 0
    training_loss = 0
    print_every = 10
    model.to(device)
    model.train()

    for e in range(epochs):
        for images, labels in trainloader:
            optimizer.zero_grad()
            steps += 1
            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

            if steps % print_every == 0:
                accuracy = 0
                validation_loss = 0
                model.eval()

                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        log_ps = model.forward(images)
                        loss = criterion(log_ps, labels)
                        validation_loss += loss.item()
                        out = torch.exp(log_ps)
                        top_p, top_c = out.topk(1, dim=1)
                        equals = labels.view(*top_c.shape) == top_c
                        accuracy += torch.mean(equals.type(torch.FloatTensor))

                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {training_loss/print_every:.3f}.. "
                      f"Validation loss: {validation_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                training_loss = 0
                model.train()
    model.to(device)
    model.eval()

    with torch.no_grad():
        test_loss = 0
        accuracy = 0

        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            test_loss += loss.item()
            out = torch.exp(log_ps)
            top_p, top_c = out.topk(1,dim=1)
            equals = top_c == labels.view(*top_c.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))

        print(f"Test loss: {test_loss/len(testloader):.3f}.. "
              f"Test accuracy: {accuracy/len(testloader):.3f}")
        
    checkpoint = checkpoint = {'state_dict': model.state_dict(),
                  'class_to_idx': class_to_idx,
                  'arch' : arch,
                  'learn_rate' : learn_rate,
                  'hidden_units' : hidden_units}
    checkpoint_path = save_dir + 'checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    
def load_model(checkpoint):
    checkpoint = torch.load(checkpoint)
    arch = checkpoint['arch']
    hidden_units = checkpoint['hidden_units']
    
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = 25088
    else:
        raise ValueError('Unsupported architecture: {}'.format(arch))
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Linear(input_size, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.5),
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def predict(image_path, checkpoint, top_k, gpu, category_names):
    
    model = load_model(checkpoint)
    pic = preprocess_image(image_path)
    
    if gpu:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = 'cpu'
        
    model.to(device)
    model.eval()
    
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None
    
    with torch.no_grad():
        pic = pic.unsqueeze(0)
        pic = pic.to(device)
        log_ps = model.forward(pic)
        out = torch.exp(log_ps)
        top_p, top_c = out.topk(top_k, dim=1)
        top_p = top_p.cpu()
        top_c = top_c.cpu()
        top_p = top_p.squeeze().numpy()
        top_c = top_c.squeeze().numpy()
        top_p = torch.tensor(top_p)

        class_to_idx = model.class_to_idx
        idx_to_class = {idx: class_ for class_, idx in class_to_idx.items()}
        if top_k == 1:
            top_c = top_c.tolist()
            top_c = [top_c]
        top_labels = [idx_to_class[idx] for idx in top_c]
        top_flower_names = [cat_to_name[c] for c in top_labels]
        
        for i in range(top_k):
            print(f'{i+1}. Class: {top_labels[i]} ({top_flower_names[i]}), Probability: {top_p[i].item():.4f}')
            