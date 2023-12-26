import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from models import MainModel
from utils import lab_to_rgb

if __name__ == '__main__':
    model = MainModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(
        torch.load(
            "final_model_weights.pt",
            map_location=device
        )
    )
    path = "Path to your black and white image"
    img = PIL.Image.open(path)
    img = img.resize((256, 256))
    img = transforms.ToTensor()(img)[:1] * 2. - 1.
    model.eval()
    with torch.no_grad():
        preds = model.net_G(img.unsqueeze(0).to(device))
    colorized = lab_to_rgb(img.unsqueeze(0), preds.cpu())[0]
    plt.imshow(colorized)