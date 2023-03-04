import torch
from torchvision import datasets, transforms
from PIL import Image

def load_data(data_dir):
    train_transforms = transforms.Compose([transforms.RandomRotation(45),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(data_dir + '/train', transform = train_transforms)
    valid_dataset = datasets.ImageFolder(data_dir + '/valid', transform = valid_transforms)
    test_dataset = datasets.ImageFolder(data_dir + '/test', transform = test_transforms)
    
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = 64, shuffle = True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size = 64)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size = 64)
    
    return trainloader, validloader, testloader, train_dataset.class_to_idx

def preprocess_image(image_dir):
    transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])
    
    image = Image.open(image_dir)
    image_tensor = transform(image)
    return image_tensor