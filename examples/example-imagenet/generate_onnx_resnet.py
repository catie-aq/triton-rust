import torch
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
resnet18 = resnet18(weights=weights, progress=False).eval()

transforms = weights.transforms()
print(transforms)

x = torch.randn(1, 3, 256, 256, requires_grad=True)
torch_out = resnet18(x)

# Export the model
torch.onnx.export(resnet18, # model being run
                  x, # model input (or a tuple for multiple inputs)
                  "resnet_18.onnx", # where to save the model (can be a file or file-like object)
                  input_names = ['input'], # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'}, # variable length axes
                                'output' : {0 : 'batch_size'}})
