from PIL import Image
import PIL
import numpy as np
from einops import rearrange, repeat


import torch
import torchvision.transforms as transforms
from diffusers import StableDiffusionPipeline

seed = 4321
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


### Stable Diffusion Models
model_id = "runwayml/stable-diffusion-v1-5" #
#model_id = "CompVis/stable-diffusion-v1-2"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to(device)
scheduler = pipe.scheduler#to(device)

# generate image from latent
vae = pipe.vae#.from_pretrained("stable-diffusion-v1-5")
vae.eval()
vae.to(device)

# encode text
tokenizer= pipe.tokenizer.from_pretrained('openai/clip-vit-large-patch14') 

text_encoder = pipe.text_encoder.from_pretrained('openai/clip-vit-large-patch14')
text_encoder.to(device)
text_encoder.eval()


# encocde image
unet = pipe.unet.to(device)
unet.eval()

del pipe



### Text Prompts
prompt = "A fantasy landscape in oil painting"
#prompt = "a number 4 in a white background"


with torch.inference_mode():
    tokens = tokenizer(prompt, padding='max_length', max_length = tokenizer.model_max_length, truncation=True, return_tensors='pt').to(device)
    text_embedding = text_encoder(tokens.input_ids)[0] #[text_encoder.output(0)]



# negative prompt. default ""
batch_size = 1
max_length = tokens.input_ids.shape[-1]
uncond_input = tokenizer([""]*batch_size, padding="max_length", max_length = tokenizer.model_max_length, return_tensors="pt").to(device)
uncond_embedding = text_encoder(uncond_input.input_ids)[0]


text_embeddings = torch.cat([uncond_embedding, text_embedding])



### Initial Image
url = "sketch-mountains-input.jpg"
init_image = Image.open(url).convert("RGB")
init_image = init_image.resize((768, 512))


w, h = init_image.size
w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
init_image = init_image.resize((w, h), resample=PIL.Image.LANCZOS)
init_image = np.array(init_image).astype(np.float32) / 255.0

init_image = init_image[None].transpose(0, 3, 1, 2)
init_image = torch.from_numpy(init_image).to(device)

init_image = repeat(init_image, '1 ... -> b ...', b=1)


### Embedd image
with torch.inference_mode():
    latents = vae.encode(init_image.to(torch.half)).latent_dist.sample()* 0.18215
    latents2 = vae.encode(init_image2.to(torch.half)).latent_dist.sample()* 0.18215
    latents = latents*0.95 + latents2*0.05


### Set noise scheduler
max_step = 6 #5 #0 # REDUCE THIS NUMBER CLOSER TO THE MAX STEP.Smaller number makes it look closer to the input image
scheduler.set_timesteps(max_step)
scheduler.timesteps


### Get latent embeddings
guidance_scale = 5
for i, t in enumerate(scheduler.timesteps):
    with torch.inference_mode():
        if i==0:
            noise = torch.randn_like(latents)
            noisey_latents = scheduler.add_noise(latents, noise, t).to(device)

        pred = unet(noisey_latents.repeat(2,1,1,1).to(torch.half), t.repeat(2).to(device), encoder_hidden_states=text_embeddings.repeat_interleave(noisey_latents.shape[0], dim=0).to(torch.half)).sample

        pred_uncond, pred_text = pred.chunk(2)
        #pred = pred_uncond + guidance_scale * (pred_uncond - pred_text)
        #pred = pred_text + guidance_scale * (pred_text-pred_uncond) #*1 + pred_uncond*0.1
        pred = pred_text
        noisey_latents = scheduler.step(pred, t, noisey_latents)['prev_sample']

torch.cuda.empty_cache() 



### Decode latent and generate image
import torchvision.transforms as T
transform_to_img = T.ToPILImage()

noisey_latents = 1 / 0.18215 * noisey_latents
with torch.inference_mode():
    dcpred = vae.decode(noisey_latents).sample

dcpred = (dcpred / 2 + 0.5).clamp(0, 1)
transform_to_img(dcpred[0])
