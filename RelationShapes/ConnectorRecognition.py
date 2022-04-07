import torch

from RelationShapes.training import Network
from torchvision.transforms import transforms
import torchvision


class ConnectorRecognition:
    def __init__(self):
        self.model = Network(8)
        self.test_transforms = transforms.Compose([
            ## color, radomrotation rotandresizecrop,
            transforms.Resize((224, 224)),
            transforms.RandomRotation(90, torchvision.transforms.InterpolationMode.BILINEAR, fill=1),
            transforms.RandomVerticalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])


    def loadModel(self):
        path = "myFirstModel.pth"
        self.model.load_state_dict(torch.load(path))

    def testBatch(self):
        from PIL import Image
        # model.pool = torch.nn.Identity()
        # get batch of images from the test DataLoader
        self.model.eval()
        # images, labels = next(iter(test_loader))
        images = Image.open('sample/image (3).png').convert('RGB')
        images = self.test_transforms(images)[None]

        # show all images as one image grid
        # imageshow(torchvision.utils.make_grid(images))

        # Show the real labels on the screen
        # print('Real labels: ', ' '.join('%5s' % classes[labels[j]]
        #                                 for j in range(batch_size)))

        # Let's see what if the model identifiers the  labels of those example
        outputs = self.model(images)

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