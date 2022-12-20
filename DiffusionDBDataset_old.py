from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import json

class DiffusionDBDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, dataset, transform=None):

        with open('WORDMAP_coco_5_cap_per_img_5_min_word_freq.json') as json_file:
            dict = json.load(json_file)

        self.imgs = dataset["train"]["image"]

        # Load encoded pompts (completely into memory)
        self.pompts = []
        self.pomptlens = []
        for prompt in dataset["train"]["prompt"]:
            for word in prompt.split():
                if word not in dict:
                    word = "<unk>"
                self.pompts.append(int(dict[word]))
            self.pomptlens.append(len(prompt.split()))

        # Load caption lengths (completely into memory)
        #self.pomptlens = [len(prompt) for prompt in self.pompts]

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform

        # Total number of datapoints
        self.dataset_size = 1000

        self.PILtransform = transforms.Compose([
            transforms.PILToTensor()
        ])

          

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        imgTensor = self.PILtransform(self.imgs[i])
        imgTensor = transforms.functional.crop(img= imgTensor, top= 0, left= 0, height= 512, width= 512)
        print(imgTensor)
        img = torch.FloatTensor(imgTensor / 255.)
        if self.transform is not None:
            img = self.transform(img)
        
        prompt = torch.LongTensor(self.pompts[i])

        pomptlen = torch.LongTensor([self.pomptlens[i]])
        print(img.shape)
        return img, prompt, pomptlen

    def __len__(self):
        return self.dataset_size