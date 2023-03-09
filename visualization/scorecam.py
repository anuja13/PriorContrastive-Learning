"""
Created on Wed Apr 29 16:11:20 2020

"""
import os
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F

from misc_functions import *


class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.network._modules.items():
            # print(module_pos, module)
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x
    
    def forward_pass_on_linear(self, x):
        '''
        Does a forward pass on Linear layers before passing to final classifier
      Args:
            x (Tensor (1,2048 )): flattened output from conv.

        Returns:
            x (Tensor (1, 128)): Input to linear classifier
        '''
        for module_pos, module in self.model.lin_head._modules.items():
            x = module(x)
        return x

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(1, -1)  # Flatten
        x = self.forward_pass_on_linear(x)
        # Forward pass on the classifier
        x = self.model.classifier(x)
        return conv_output, x


class ScoreCam():
    """
        Produces class activation map
    """
    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 3)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
            # print(target_class, model_output.data.numpy())
        # Get convolution outputs
        target = conv_output[0]
        # Create empty numpy array for cam
        cam = np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i in range(len(target)):
            # Unsqueeze to 4D
            saliency_map = torch.unsqueeze(torch.unsqueeze(target[i, :, :],0),0)
            # Upsampling to input size
            saliency_map = F.interpolate(saliency_map, size=(224, 224), mode='bilinear', align_corners=False) # this was 224,224
            if saliency_map.max() == saliency_map.min():
                continue
            # Scale between 0-1
            norm_saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
            # Get the target score
            w = F.softmax(self.extractor.forward_pass(input_image*norm_saliency_map)[1],dim=1)[0][target_class]
            cam += w.data.numpy() * target[i, :, :].data.numpy()
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        cam = np.uint8(Image.fromarray(cam).resize((input_image.shape[2],
                       input_image.shape[3]), Image.ANTIALIAS))/255
        return cam, target_class, model_output.data.numpy()
    
    def generate_class_label(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 3)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        output = F.softmax(model_output, dim=1)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
            print(target_class, output[0])
        
        return  target_class, output[0][target_class].item()

if __name__ == '__main__':
    # Get params
    PGCon_giana_pretrained_path =  '/home/user1/checkpoint/epoch_329'
    path2images = '/home/user1/images'

    for i, image in enumerate(sorted(os.listdir(path2images))):
        img_path = os.path.join(path2images,image)    
        # img_path = example_list[example_index][0]
        target_class = None
        file_name_to_export = str(i)+img_path[img_path.rfind('/')+1:img_path.rfind('.')]
        # Read image
        original_image = Image.open(img_path).convert('RGB')
        original_image = original_image.crop((50,50,500,500))
        resize_original_image = original_image.resize((224, 224), Image.ANTIALIAS)
        # Process image
        prep_img = preprocess_image(original_image)
        # Load model
        pretrained_model = Network()
        state_dict = torch.load(PGCon_giana_pretrained_path)
        # unparallel dict
        new_state_dict = collections.OrderedDict()
        for key, value in state_dict['model'].items():
            new_key = transfom_key(key)
            new_state_dict[new_key] = value
        pretrained_model.load_state_dict(new_state_dict)
    
        # Score cam
        score_cam = ScoreCam(pretrained_model, target_layer=7)
        # Generate cam mask
        # cam, pred_class, score = score_cam.generate_cam(prep_img, target_class)
        pred_class, score = score_cam.generate_class_label(prep_img, target_class)
        # Save mask
        # save_class_activation_images(resize_original_image, cam,pred_class, score,  file_name_to_export)
        save_noisy_labels(resize_original_image,pred_class, score,  file_name_to_export)
        print('saved for image :',i)
    print('Score cam completed')
