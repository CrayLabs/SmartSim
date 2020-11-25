from skimage import io
import json
import torch
print("USING CUDA", torch.cuda.is_available())

class_idx = json.load(open("./imagenet_classes.json"))
filepath = './cat.jpg'

model = torch.jit.load('resnet50.pt').cuda()
model.to(torch.device("cuda"))


@torch.jit.script
def pre_process(image):
    # todo: Is there any way we could avoid redefining these
    mean = torch.zeros(3).float()
    std = torch.zeros(3).float()
    # how to remove this hack
    mean[0], mean[1], mean[2] = 0.485, 0.456, 0.406
    std[0], std[1], std[2] = 0.229, 0.224, 0.225
    mean = mean.unsqueeze(1).unsqueeze(1)
    std = std.unsqueeze(1).unsqueeze(1)
    temp = image.float().div(255).permute(2, 0, 1)
    return temp.sub(mean).div(std).unsqueeze(0)


@torch.jit.script
def post_process(output):
    return output.max(1)[1]


numpy_img = io.imread(filepath)
image = torch.from_numpy(numpy_img)
batch = pre_process(image).cuda()
batch.to(torch.device("cuda"))
out = model(batch)
print(out.shape)
print(out)
ind = post_process(out).cpu()
print(ind.item(), class_idx[str(ind.item())])
