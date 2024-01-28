from torch import nn
import torchvision

def choice(model_name='regnet_y_16gf', finetuning_type="Finetuning", pretrained=True, device='cuda:0'):

    if model_name == "ResNet18":
        model = torchvision.models.resnet18(pretrained=pretrained)
        model.fc = nn.Sequential(
            nn.Linear(512, 100),
            nn.ReLU(),
            nn.Linear(100, 40),
            nn.ReLU(),
            nn.Linear(40, 9))
        if finetuning_type == "Only Head":
            for param in model.parameters():
                param.requires_grad = False 
            for param in model.fc.parameters():
                param.requires_grad = True
        elif finetuning_type == "Finetuning":
            for param in model.parameters():
                param.requires_grad = True
    elif model_name == "resnext50_32x4d":
        model = torchvision.models.resnext50_32x4d(pretrained=pretrained)

        model.fc = nn.Sequential(
        
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, 40),
        nn.ReLU(),
        nn.Linear(40, 9))
        if finetuning_type == "Only Head":
            for param in model.parameters():
                param.requires_grad = False 
            for param in model.fc.parameters():
                param.requires_grad = True
        elif finetuning_type == "Finetuning":
            for param in model.parameters():
                param.requires_grad = True
    elif model_name == "regnet_y_16gf":
        model = torchvision.models.regnet_y_16gf(weights=torchvision.models.RegNet_Y_16GF_Weights.IMAGENET1K_SWAG_E2E_V1)
        model.fc = nn.Sequential(
        nn.Linear(3024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Linear(512, 100),
        nn.ReLU(),
        nn.Linear(100, 40),
        nn.ReLU(),
        nn.Linear(40, 9))
        if finetuning_type == "Only Head":
            for param in model.parameters():
                param.requires_grad = False 
            for param in model.fc.parameters():
                param.requires_grad = True
        elif finetuning_type == "Finetuning":
            for param in model.parameters():
                param.requires_grad = True
    elif model_name == "mobilenet_v3_small":
        model = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
        model.classifier = nn.Sequential(
            nn.Linear(576, 1024),
            nn.Hardswish(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64,9)
            )
        if finetuning_type == "Only Head":
            for param in model.parameters():
                param.requires_grad = False 
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif finetuning_type == "Finetuning":
            for param in model.parameters():
                param.requires_grad = True
    elif model_name == "efficientnet_v2_s":
        model = torchvision.models.efficientnet_v2_s(pretrained=pretrained).to(device)
        model.classifier = nn.Sequential(
            nn.Linear(1280, 512),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64,9)
            )
        if finetuning_type == "Only Head":
            for param in model.parameters():
                param.requires_grad = False 
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif finetuning_type == "Finetuning":
            for param in model.parameters():
                param.requires_grad = True
    return model