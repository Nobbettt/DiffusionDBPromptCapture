import torchvision.transforms as transforms

def tensor_to_img(tensor):
    transform = transforms.ToPILImage()
    img = transform(tensor)
    img.show()