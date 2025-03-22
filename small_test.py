from proxy_net.resnet import Resnet18
import torch
model = Resnet18(embedding_size=512, bg_embedding_size=1024, pretrained=True, is_norm=True, is_student=True, bn_freeze=False).cuda()
x = torch.randn(1, 3, 32, 32).cuda()
logits, features = model(x)
print("Logits shape:", logits.shape)
print("Features shape:", features.shape)
