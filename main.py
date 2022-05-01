#%%
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
from PIL import Image
import torch.optim as optim
import torchvision
import sys
import os
import torch.nn.functional as F

#%%
#model = models.vgg19(pretrained=True).features
class VGG(nn.Module):
  def __init__(self):
    super(VGG, self).__init__()

    self.chosen_features = ["0", "5", "10", "19", "28"]
    self.model = models.vgg19(pretrained=True).features[:29]

  def forward(self, x):
    features = []

    for layer_num, layer in enumerate(self.model):
      x = layer(x)

      if str(layer_num) in self.chosen_features:
        features.append(x)

    return features
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
imsize = 356
loader = transforms.Compose(
  [
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
  ]
)

transform = transforms.Compose(
  [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])



def load_image(image_name):
  image = Image.open(image_name)
  image = loader(image).unsqueeze(0)
  return image.to(device)


count = 0
# style_img = load_image("Style.jpg")
model = VGG().to(device).eval()
total_steps = 500
learning_rate = 0.3
alpha = 1
beta = 0.02

def perform_style(images,out_location,style_img):
  global count
  original_img = [transforms.Resize(size=size)(images) for size in (1, 3, 356)][2]
  generated = original_img.clone().requires_grad_(True)

  # original_img = load_image("Original.png")
  # generated = original_img.clone().requires_grad_(True)

  optimizer = optim.Adam([generated], lr=learning_rate)

  for step in range(total_steps):
    # Obtain the convolution features in specifically chosen layers
    generated_features = model(generated)
    original_img_features = model(original_img)
    style_features = model(style_img)

    # Loss is 0 initially
    style_loss = original_loss = 0

    # iterate through all the features for the chosen layers
    for gen_feature, orig_feature, style_feature in zip(
        generated_features, original_img_features, style_features
      ):
      # batch_size will just be 1
      batch_size, channel, height, width = gen_feature.shape
      original_loss += torch.mean((gen_feature - orig_feature) ** 2)
      print(f"og loss: {original_loss}")
      # Compute Gram Matrix of generated
      G = gen_feature.view(channel, height * width).mm(
        gen_feature.view(channel, height * width).t()
      )
      # Compute Gram Matrix of Style
      A = style_feature.view(channel, height * width).mm(
        style_feature.view(channel, height * width).t()
      )
      style_loss += torch.mean((G - A) ** 2)
      print(f"style loss: {style_loss}\n")

    total_loss = alpha * original_loss + beta * style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

  print(total_loss)
  if not os.path.isdir(out_location):
      os.mkdir(out_location)
  name = out_location + str(count) + "generated.png"
  save_image(generated, name)
  name = out_location + str(count) + "original.png"
  save_image(original_img, name)
  count += 1

def perform_styles(trainloader,out_location,style_img):
  for i, data in enumerate(trainloader, 0):
    images, labels = data
    images = images.cuda()
    perform_style(images,out_location,style_img)


if __name__ ==  '__main__':
  out_location = "Output/"
  if sys.argv[1] == "random":
    out_location += "random/"
    try:
      og_img = load_image(sys.argv[2])
      style_image  = load_image(sys.argv[3])
    except:
      print("Filure loading original or style image")
      sys.exit(1)
    perform_style(og_img,out_location,style_image)
    sys.exit(0)
      
  else:
    if sys.argv[1] == 'cifar10':
      trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
        download=True, transform=transform)
      out_location += "cifar10/"
    elif sys.argv[1] == 'cifar100':
      trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
        download=True, transform=transform)
      out_location += "cifar100/"
    try:
        style_image  = load_image(sys.argv[2])
    except:
      print("Style image input error")
      sys.exit(1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
      shuffle=True, num_workers=2)
    perform_styles(trainloader,out_location,style_image)
    sys.exit(0)