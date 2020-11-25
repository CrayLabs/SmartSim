import torchvision.models as models
import torch
print("USING CUDA", torch.cuda.is_available())

model = models.resnet101(pretrained=True)
model.cuda()
model.to(torch.device("cuda"))
model.eval()

batch = torch.randn((1, 3, 224, 224)).cuda()
batch.to(torch.device("cuda"))
traced_model = torch.jit.trace(model, batch)
torch.jit.save(traced_model, 'resnet101.pt')
