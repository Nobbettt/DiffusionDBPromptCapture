from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import numpy as np

class DiffusionDBDataLoader(Dataset):
    def __init__(self, images, prompts, max_img_dim, word_map_dict, batch_size, transform=None, max_length = None, remove_unk = False):
        print("Assertions and dimensions")
        assert len(images) == len(prompts)
        self.max_img_width = max_img_dim[0]
        self.max_img_height = max_img_dim[1]
        self.canvas = torch.full((3,self.max_img_height,self.max_img_width),1.)
        self.batch_size = batch_size
        
        # Load encoded pompts (completely into memory)
        print("Loading prompts")
        pompts = []
        pomptlens = []
        
        for prompt in prompts:
            prompt_encoding = []
            prompt_encoding.append(int(word_map_dict["<start>"]))
            
            for word in prompt.split():
                if word not in word_map_dict:
                    if remove_unk:
                        continue
                    
                    word = "<unk>"
                prompt_encoding.append(int(word_map_dict[word]))
                if len(prompt_encoding) >= max_length:
                    break
            
            prompt_encoding.append(int(word_map_dict["<end>"])) 
            
            pompts.append(prompt_encoding)
            pomptlens.append(len(prompt_encoding))
        
        self.max_encoded_prompt_length = max(pomptlens)

        pompts = [torch.LongTensor(p + [0]*(self.max_encoded_prompt_length-len(p))) for p in pompts]

        print("Setting up transformer")
        self.transform = transform

        self.dataset_size = len(images)
        
        self.PILtransform = transforms.Compose([
            transforms.PILToTensor()
        ])
        
        print("Creating batches")
        self.batch_images = []
        self.batch_prompt = []
        self.batch_pomptlens = []
        
        add_val = 1
        if self.dataset_size % batch_size == 0:
            add_val = 0
        
        for i in range((self.dataset_size//batch_size)+add_val):
            start_index = min(i*self.batch_size, self.dataset_size)
            end_index = min((i+1)*self.batch_size, self.dataset_size)

            self.batch_images.append(images[start_index:end_index])
            self.batch_prompt.append(pompts[start_index:end_index])
            self.batch_pomptlens.append(pomptlens[start_index:end_index])


    def __len__(self):
        return self.dataset_size

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image

        batch_images = self.batch_images[i]
        
        for j in range(len(batch_images)):
            
            if torch.is_tensor(batch_images[j]):
                continue
            
            batch_images[j] = self.PILtransform(batch_images[j])
            
            if (batch_images[j].shape[1] > self.max_img_height):
                batch_images[j] = transforms.functional.crop(img=batch_images[j], top= int(round((batch_images[j].shape[1] - self.max_img_height)/2)), left= 0, height= self.max_img_height, width= batch_images[j].shape[2])
            
            if (batch_images[j].shape[2] > self.max_img_width):
                batch_images[j] = transforms.functional.crop(img=batch_images[j], top= 0, left= int(round((batch_images[j].shape[2] - self.max_img_width)/2)), height= batch_images[j].shape[1], width= self.max_img_width)

            batch_images[j] = torch.FloatTensor(batch_images[j] / 255.)
            
            if self.transform is not None:
                batch_images[j] = self.transform(batch_images[j])
            
            batch_images[j] = self.pad_img(batch_images[j])
        
        batch_images = torch.stack(batch_images)

        prompt = torch.stack(self.batch_prompt[i])
        
        pomptlen = torch.LongTensor(self.batch_pomptlens[i])

        return batch_images, prompt, pomptlen

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

