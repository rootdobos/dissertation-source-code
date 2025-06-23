from torchvision import transforms
from torchvision.models.resnet import ResNeXt50_32X4D_Weights
from torchvision.models.efficientnet import  EfficientNet_B4_Weights, EfficientNet_B3_Weights, EfficientNet_V2_S_Weights

def transform_resnet(image):
    preprocess= transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)

def transform_resnext(image):
    weights = ResNeXt50_32X4D_Weights.IMAGENET1K_V2
    transform = weights.transforms()
    return transform(image)

def transform_efficientNet(image):
    preprocess= transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return preprocess(image)

def transform_efficientNetB4(image):
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    return transform(image)

def transform_efficientNetB3(image):
    weights = EfficientNet_B3_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    return transform(image)

def transform_efficientNetV2(image):
    weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1
    transform = weights.transforms()
    return transform(image)