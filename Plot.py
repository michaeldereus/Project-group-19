from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

from torchvision.utils import make_grid
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

loader = transforms.Compose(
    [
        transforms.Resize((356, 356)),
        transforms.ToTensor()
    ]
)

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)

def show_batch(dl):
    for i in range(3):
        fig, ax = plt.subplots(figsize=(5, 10))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(dl[i]).permute(1, 2, 0))
    plt.show()

#main = Image.open('Orignal.png')
#style = Image.open('Style.jpg')
#res = Image.open('Output/59generated.png')
#images_2=[main,style,res]
original_img=load_image("Original.png")
style_img=load_image("Style.jpg")
generated = load_image("Output/15generated.png")

images=[original_img.to('cpu'),style_img.to('cpu'),generated.to('cpu').detach()]
show_batch(images)