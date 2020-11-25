from skimage import io
import json
import torch

"""
Script to check the preprocessing function and
the JIT inference of the model we have saved
"""

@torch.jit.script
def pre_process_3ch(image):
    mean = torch.zeros(1).float().to(image.device)
    std = torch.zeros(1).float().to(image.device)
    mean[0] = 0.1307
    std[0] = 0.3081
    mean = mean.unsqueeze(1).unsqueeze(1)
    std = std.unsqueeze(1).unsqueeze(1)
    temp = image.float().div(28).permute(1, 0)
    return temp.sub(mean).div(std).unsqueeze(0)


# load the image to check with and preprocess
# it with the JIT function

# do it with a 7
#filepath = '../data/seven.png'
#numpy_img = io.imread(filepath, as_gray=True)

# do it with a 1
filepath = '../data/one.png'
numpy_img = io.imread(filepath)

image = torch.from_numpy(numpy_img)
batch = pre_process_3ch(image).cuda()

# load in the model and perform the jit inference
model = torch.jit.load('mnist_cnn.pt')
with torch.no_grad():
    out = model(batch)

ps = torch.exp(out).cpu()
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
