"""
Reference from
https://github.com/CompVis/stable-diffusion/tree/main
"""
from PIL import Image
import PIL
import numpy as np
from einops import rearrange, repeat


import torch
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline
import torchvision.transforms as T

seed = 4321
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'



class StableDiffusionAdaptivePainter():
    def __init__(self, batch_size, max_step = 20, model_id="runwayml/stable-diffusion-v1-5"):
        self.batch_size = batch_size
        self.max_step = max_step 

        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        
        self.scheduler = pipe.scheduler
        self.set_scheduler()

        self.tokenizer= pipe.tokenizer.from_pretrained('openai/clip-vit-large-patch14') 
        self.text_encoder = pipe.text_encoder.from_pretrained('openai/clip-vit-large-patch14')
        self.text_encoder.to(device)
        self.text_encoder.eval()

        self.vae = pipe.vae
        self.vae.eval()
        self.vae.to(device)

        self.unet = pipe.unet
        self.unet.eval()
        self.unet.to(device)

        self.transform_to_img = T.ToPILImage()

    def get_text_embeddings(self, prompt, neg_prompt=""):
        tokens = self.tokenizer(prompt, padding='max_length', max_length = self.tokenizer.model_max_length, truncation=True, return_tensors='pt').to(device)
        text_embedding = self.text_encoder(tokens.input_ids)[0] #[text_encoder.output(0)]

       
        max_length = tokens.input_ids.shape[-1]
        uncond_input = self.tokenizer([neg_prompt]*self.batch_size, padding="max_length", max_length = self.tokenizer.model_max_length, return_tensors="pt").to(device)
        uncond_embedding = self.text_encoder(uncond_input.input_ids)[0]

        text_embeddings = torch.cat([uncond_embedding, text_embedding])
        return text_embeddings

    def preprocess_image(self, image):    
        # image is PIL image
        image = image.convert("RGB")
        w, h = init_image.size
        w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
        image = image.resize((w, h), resample=PIL.Image.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0

        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image).to(device)

        image = repeat(image, '1 ... -> b ...', b=self.batch_size)
        return image

    def get_image_embeddings(self, current_image, target_image):
        curr_image_input = self.preprocess_image(current_image)
        target_image_input = self.preprocess_image(target_image)

        curr_latents = vae.encode(curr_image_input.to(torch.half)).latent_dist.sample()* 0.18215
        target_latents = vae.encode(target_image_input.to(torch.half)).latent_dist.sample()* 0.18215
        latents = curr_latents*0.85 + target_latents*0.15


    def set_scheduler(self):
        self.scheduler.set_timesteps(self.max_step)
    
    def predict_latents(self, image_embeddings, text_embeddings):
        for i, t in enumerate(scheduler.timesteps):
            with torch.inference_mode():
                if i==0:
                    noise = torch.randn_like(image_embeddings)
                    noisey_latents = scheduler.add_noise(image_embeddings, noise, t).to(device)

                pred = unet(noisey_latents.repeat(2,1,1,1).to(torch.half), t.repeat(2).to(device), encoder_hidden_states=text_embeddings.repeat_interleave(noisey_latents.shape[0], dim=0).to(torch.half)).sample

                pred_uncond, pred_text = pred.chunk(2)
                #pred = pred_uncond + guidance_scale * (pred_uncond - pred_text)
                #pred = pred_text + guidance_scale * (pred_text-pred_uncond) #*1 + pred_uncond*0.1
                pred = pred_text
                noisey_latents = scheduler.step(pred, t, noisey_latents)['prev_sample']

        return noisey_latents 

    def generate_image_from_predictions(self, prediction):
        prediction = 1 / 0.18215 * prediction
        with torch.inference_mode():
            output_image = vae.decode(prediction).sample
        output_image = (output_image / 2 + 0.5).clamp(0, 1)
        return output_image


    def generate_image(self, prompt, sketch, target_image):

        text_embeddings = self.get_text_embeddings(prompt)
        image_embeddings = self.get_image_embeddings(sketch, target_image)

        prediction = self.predict_latents(image_embeddings, text_embeddings)
        output = self.generate_image_from_predictions(prediction)

        output_image = self.transform_to_img(output[0])
        return output_image #PIL Image format