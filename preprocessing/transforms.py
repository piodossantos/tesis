from torchvision import transforms


BASELINE = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(120),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

YOLO_BASELINE = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize([640, 640]),
    transforms.ToTensor(),
])