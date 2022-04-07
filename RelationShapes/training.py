import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.autograd import Variable

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class Dataset(torch.utils.data.Dataset):
    def __int__(self, path_to_data, transform=None):
        root = Path(path_to_data)

        self.images = list(root.iterdir())

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        if self.transform:
            img = self.transform(img)

        return img


# Loading and normalizing the data.
# Define transformations for the training and test sets
transformations = transforms.Compose([
    ## color, radomrotation rotandresizecrop,
    transforms.Resize((224, 224)),
    transforms.RandomRotation(90, torchvision.transforms.InterpolationMode.BILINEAR, fill=1),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

test_transforms = transforms.Compose([
    ## color, radomrotation rotandresizecrop,
    transforms.Resize((224, 224)),
    transforms.RandomRotation(90, torchvision.transforms.InterpolationMode.BILINEAR, fill=1),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# CIFAR10 dataset consists of 50K training images. We define the batch size of 10 to load 5,000 batches of images.
batch_size = 1
number_of_labels = 8

# Create an instance for training.
# When we run this code for the first time, the CIFAR10 train dataset will be downloaded locally.
# train_set =CIFAR10(root=,train=True,transform=transformations,download=True)
path_to_data = "img"
train_set = torchvision.datasets.ImageFolder(path_to_data, transform=transformations)

# Create a loader for the training set which will read the data within batch size and put into memory.
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
print("The number of images in a training set is: ", len(train_loader) * batch_size)

# Create an instance for testing, note that train is set to False.
# When we run this code for the first time, the CIFAR10 test dataset will be downloaded locally.
# test_set = CIFAR10(root="train", train=False, transform=transformations, download=True)
test_set = torchvision.datasets.ImageFolder('img', transform=transformations)

# Create a loader for the test set which will read the data within batch size and put into memory.
# Note that each shuffle is set to false for the test loader.
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
print("The number of images in a test set is: ", len(test_loader) * batch_size)

print("The number of batches per epoch is: ", len(train_loader))
classes = ('solid_diamonds', 'solid_solid_triangles, solid_dotted_diamonds',
           'solid_dotted_triangles', 'hollow_triangles', 'hollow_diamonds',
           'hollow_dotted_triangles', 'hollow_dotted_diamonds')


# Define a convolution neural network
class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.pool = nn.AdaptiveAvgPool2d(1) ##
        # self.fc1 = nn.Conv2d(in_channels=24, out_channels=num_classes, kernel_size=1, stride=1, padding=1)

        self.fc1 = nn.Linear(24, num_classes)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))
        output = F.relu(self.bn2(self.conv2(output)))
        output = self.pool(output)
        output = F.relu(self.bn4(self.conv4(output)))
        output = F.relu(self.bn5(self.conv5(output)))
        output = self.pool(output)
        output = output.view(-1, 24)
        output = self.fc1(output)

        return output


# Instantiate a neural network model
model = Network(8)

# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)


# Function to save the model
def saveModel():
    path = "./myFirstModel.pth"
    torch.save(model.state_dict(), path)


# Function to test the model with the test dataset and print the accuracy for the test images
def testAccuracy():
    model.eval()
    accuracy = 0.0
    total = 0.0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return accuracy


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("The model will be running on", device, "device")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):

            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()  # extract the loss value
            if i % 1000 == 999:
                # print every 1000 (twice per epoch)
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = testAccuracy()
        print('For epoch', epoch + 1, 'the test accuracy over the whole test set is %d %%' % (accuracy))

        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            saveModel()
            best_accuracy = accuracy


# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    from PIL import Image
    # model.pool = torch.nn.Identity()
   # get batch of images from the test DataLoader
    model.eval()
    # images, labels = next(iter(test_loader))
    images = Image.open('sample/image (3).png').convert('RGB')
    images = test_transforms(images)[None]

    # show all images as one image grid
    # imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    # print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
    #                                 for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    if _.detach().numpy()[0] > 20:
        predicted_class = train_set.classes[outputs.argmax().item()]
    else:
        predicted_class = 'none'

    print(f'Predicted class: {predicted_class}')


    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                                  for j in range(batch_size)))


# Function to test what classes performed well
# def testClassess():
#     class_correct = list(0. for i in range(number_of_labels))
#     class_total = list(0. for i in range(number_of_labels))
#     with torch.no_grad():
#         for data in test_loader:
#             images, labels = data
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)
#             c = (predicted == labels).squeeze()
#             for i in range(batch_size - 1):
#                 label = labels[i]
#                 class_correct[label] += c[i].item()
#                 class_total[label] += 1
#
#     for i in range(number_of_labels):
#         print('Accuracy of %5s : %2d %%' % (
#             classes[i], 100 * class_correct[i] / class_total[i]))

# def test_image ():
#     outputs = model('img/hollow_triangles/1.png')
#
#     return 'label'

if __name__ == "__main__":
    # Let's build our model
    # train(8)
    print('Finished Training')

    # Test which classes performed well
    # testModelAccuracy()

    # Let's load the model we just created and test the accuracy per label
    model = Network(8)
    path = "myFirstModel.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()
    # testClassess()
