import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms

batch_size = 1
number_of_labels = 8
test_transforms = transforms.Compose([
    ## color, radomrotation rotandresizecrop,
    transforms.Resize((224, 224)),
    transforms.RandomRotation(90, torchvision.transforms.InterpolationMode.BILINEAR, fill=1),
    transforms.RandomVerticalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

classes = sorted(('solid_diamonds', 'solid_solid_triangles', 'solid_dotted_diamonds',
                  'solid_dotted_triangles', 'hollow_triangles', 'hollow_diamonds',
                  'hollow_dotted_triangles', 'hollow_dotted_diamonds'))



def initialize():
    model = Network(8)
    path = "RelationShapes/myFirstModel.pth"
    model.load_state_dict(torch.load(path))

    return model


def recognize(img):
    from PIL import Image
    model = initialize()
    # model.pool = torch.nn.Identity()
    model.eval()
    im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    images = test_transforms(im_pil)[None]

    # show all images as one image grid
    # imageshow(torchvision.utils.make_grid(images))

    # Show the real labels on the screen
    # print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
    #                                 for j in range(batch_size)))

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    score, predicted = torch.max(outputs.softmax(1), 1)
    if score.detach().numpy()[0] > 0.8:
        predicted_class = classes[outputs.argmax().item()]
    else:
        predicted_class = 'none'

    # print(f'Predicted class: {predicted_class}')
    return predicted_class

    # Let's show the predicted labels on the screen to compare with the real ones
    # print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
    #                               for j in range(batch_size)))

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
        self.pool = nn.AdaptiveAvgPool2d(1)  ##
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
