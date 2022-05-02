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
import matplotlib.pyplot as plt
import numpy as np

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
learning_rate = 0.05
alpha = 1
beta = 0.02
loss_hist = []

def graph_loss(loss_hist, out_location, style_img_name):
      # plt.style.use('ggplot')
  # print(loss_hist)
  style_img_name = style_img_name.split('.')[0]
  save_name = f"{out_location}/{style_img_name}_loss.png"
  
  x = []
  y = []
  i = 0
  
  for i, loss in enumerate(loss_hist):
    x.append(i)
    y.append(loss)
      
  fig, ax = plt.subplots()

  ax.plot(x , y, linewidth=2)
  ax.set(xlim=(0, total_steps), ylim=(0, 100000), xlabel="Epoch", ylabel="Loss", 
      title=style_img_name, xticks=np.arange(0,total_steps,50))
  fig1 = plt.gcf()
  plt.show()
  plt.draw()
  fig1.savefig(save_name)

def perform_style(images,out_location,style_img, style_img_name):
  global count
  
  original_img = [transforms.Resize(size=size)(images) for size in (1, 3, 356)][2]
  generated = original_img.clone().requires_grad_(True)

  # original_img = load_image("Original.png")
  # generated = original_img.clone().requires_grad_(True)

  optimizer = optim.Adam([generated], lr=learning_rate)
  print(steps)
  for step in range(steps):
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
      # Compute Gram Matrix of generated
      G = gen_feature.view(channel, height * width).mm(
        gen_feature.view(channel, height * width).t()
      )
      # Compute Gram Matrix of Style
      A = style_feature.view(channel, height * width).mm(
        style_feature.view(channel, height * width).t()
      )
      style_loss += torch.mean((G - A) ** 2)

    total_loss = alpha * original_loss + beta * style_loss
    loss_hist.append(float(f"{total_loss}"))
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
  graph_loss(loss_hist, out_location, style_img_name)
    
  if not os.path.isdir(out_location):
    os.mkdir(out_location)
  name = out_location + str(count) + "generated.png"
  save_image(generated, name)
  name = out_location + str(count) + "original.png"
  save_image(original_img, name)
  count += 1
  
  
  
  
def perform_styles(trainloader,out_location,style_img,total_steps, style_img_name):
  for i, data in enumerate(trainloader, 0):
    images, labels = data
    images = images.cuda()
    perform_style(images,out_location,style_img,total_steps)

if __name__ ==  '__main__':
  total_steps = 200
  out_location = "Output/"
  if "style-level" in sys.argv:
      level = sys.argv.index("style-level") + 1

      if sys.argv[level] == '1':
          total_steps = 10
      elif sys.argv[level] == '2':
          total_steps = 300
      elif sys.argv[level] == '3':
          total_steps = 900
          
      print(level, total_steps)

  if sys.argv[1] == "random":
    out_location += "random/"
    try:
      og_img = load_image(sys.argv[2])
      style_name = sys.argv[3]
      style_image  = load_image(style_name)
    except:
      print("Filure loading original or style image")
      sys.exit(1)
    perform_style(og_img,out_location,style_image,total_steps,style_name)
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
      style_name = sys.argv[2]
      style_image  = load_image(style_name)
    except:
      print("Style image input error")
      sys.exit(1)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1,
      shuffle=True, num_workers=2)
    perform_styles(trainloader,out_location,style_image,total_steps, style_name)
    # graph_loss(loss_hist, total_steps, style_name, out_location)
    sys.exit(0)
  
  
