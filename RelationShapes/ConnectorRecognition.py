import cv2
import torch
import torchvision
from torchvision.transforms import transforms

from RelationShapes.training import Network

batch_size = 1
number_of_labels = 8
test_transforms = transforms.Compose([
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
    model.eval()
    im_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    images = test_transforms(im_pil)[None]

    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)

    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    score, predicted = torch.max(outputs.softmax(1), 1)

    score, predicted = torch.max(outputs, 1)
    if score.data.numpy()[0] > 25:
        predicted_class = classes[outputs.argmax().item()]
    else:
        predicted_class = 'none'
    return predicted_class


