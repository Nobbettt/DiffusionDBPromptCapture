from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms

class DiffusionDBDataLoader(Dataset):
    def __init__(self, images, prompts, max_img_dim, word_map_dict, transform=None):
        assert len(images) == len(prompts)
        max_img_width, max_img_height = max_img_dim
        self.canvas = torch.full((3,max_img_height,max_img_width),1.)
        self.images = images

        # Load encoded pompts (completely into memory)
        self.pompts = []
        self.pomptlens = []
        for prompt in prompts:
            prompt_encoding = []
            for word in prompt.split():
                if word not in word_map_dict:
                    word = "<unk>"
                prompt_encoding.append(int(word_map_dict[word]))
            self.pompts.append(prompt_encoding)
            self.pomptlens.append(len(prompt_encoding))
        
        self.transform = transform

        self.dataset_size = len(images)

        self.PILtransform = transforms.Compose([
            transforms.PILToTensor()
        ])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        image = self.PILtransform(self.images[i])
        #image = transforms.functional.crop(img= image, top= 0, left= 0, height= 512, width= 512)
        image = image

        image = torch.FloatTensor(image / 255.)
        
        if self.transform is not None:
            image = self.transform(image)
        
        image = self.pad_img(image)

        prompt = torch.LongTensor(self.pompts[i])

        pomptlen = torch.LongTensor([self.pomptlens[i]])

        return image, prompt, pomptlen

    def pad_img(self, img):
        canvas = self.canvas
        x_pos = int((canvas.shape[2] - img.shape[2]) / 2)
        y_pos = int((canvas.shape[1] - img.shape[1]) / 2)

        channels = list(range(canvas.shape[-3]))

        img_height, img_width = img.shape[-2], img.shape[-1]

        height = img_height if y_pos + img_height < canvas.shape[-2] else canvas.shape[-2] - y_pos
        width = img_width if x_pos + img_width < canvas.shape[-1] else canvas.shape[-1] - x_pos

        if len(img.shape) == 3  and canvas.shape[-3] == img.shape[-3]:
            canvas[..., channels, y_pos:y_pos+height, x_pos:x_pos+width] = img[channels, :height, :width]  
        else:
            canvas[..., channels, y_pos:y_pos+height, x_pos:x_pos+width] = img[..., :height, :width]
        
        return canvas

