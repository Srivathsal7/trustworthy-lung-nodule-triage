import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MCDropout(nn.Dropout):
    """Monte Carlo Dropout layer that stays active even in eval mode."""
    def forward(self, input_tensor):
        return nn.functional.dropout(input_tensor, p=self.p, training=True, inplace=self.inplace)

class NoduleResNet(nn.Module):
    def __init__(self, num_classes=2, dropout_rate=0.3):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT
        self.backbone = resnet18(weights=weights)
        num_feats = self.backbone.fc.in_features
        
        # Replace classification head with MC Dropout + Linear
        self.backbone.fc = nn.Sequential(
            MCDropout(p=dropout_rate),
            nn.Linear(num_feats, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

    def get_last_conv_layer(self):
        # Target layer for Grad-CAM explainability
        return self.backbone.layer4[-1].conv2

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print(f"MODEL VERIFICATION ON DEVICE: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = NoduleResNet(num_classes=2, dropout_rate=0.3).to(device)
    dummy_input = torch.randn(8, 3, 64, 64, device=device)
    output = model(dummy_input)
    print(f"Forward Pass Successful!")
    print(f"  • Batch Input Shape:  {dummy_input.shape}")
    print(f"  • Output Logits Shape: {output.shape}")
    print("=" * 65)