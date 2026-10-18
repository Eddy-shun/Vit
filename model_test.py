import torch
import torch.utils.data as data
from torchvision import transforms
from main import VisionTransformer
from torchvision.datasets import ImageFolder
from PIL import Image

def test_data():
    path = r'dataset\val'
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    test_data = ImageFolder(path, transform=test_transform)

    test_loader = data.DataLoader(test_data, batch_size=1, shuffle=True, num_workers=0)

    return test_loader

def test_model(model, test_loader):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    test_correct = 0.0
    test_num = 0

    with torch.no_grad():
        for data_x, data_y in test_loader:
            data_x = data_x.to(device)
            data_y = data_y.to(device)

            model.eval()
            output = model(data_x)

            pre_lab = torch.argmax(output, dim=1)
            test_correct += torch.sum((pre_lab == data_y.data))
            test_num += data_x.size(0)

        test_acc_all = test_correct.double().item() / test_num
        print("Test Accuracy: {}".format(test_acc_all))

if __name__ == '__main__':
    model = VisionTransformer()
    model.load_state_dict(torch.load('best_model.pth'))
    test_loader = test_data()
    test_model(model, test_loader)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    classes = ['cat', 'dog']

    image = Image.open('dog.41.jpg')
    test_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image = test_transform(image)

    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        image = image.to(device)
        output = model(image)
        pre_lab = torch.argmax(output, dim=1)
        result = pre_lab.item()
    print('The result is', classes[result])