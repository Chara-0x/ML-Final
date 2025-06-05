import struct
import numpy as np
import json
import sys
import torch
from PIL import Image
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDIMScheduler,
    DDIMInverseScheduler
)
from transformers import CLIPTextModel, CLIPTokenizer

MODEL    = "CompVis/stable-diffusion-v1-4"
W, H     = 96, 96
STEPS    = 50
GUIDANCE  = 7.0

goal = (
    Image.open(sys.argv[1])
         .convert("RGB")
         .resize((W, H))
)
goal_np = np.array(goal, dtype=np.float32) / 255 * 2 - 1

tok   = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True)
clip  = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", local_files_only=True, torch_dtype=torch.float32)
unet  = UNet2DConditionModel.from_pretrained(MODEL, subfolder="unet", torch_dtype=torch.float32)
vae   = AutoencoderKL.from_pretrained(MODEL, subfolder="vae", torch_dtype=torch.float32)


scheduler = DDIMScheduler(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False,
    set_alpha_to_one=False,
)
scheduler.set_timesteps(STEPS)

inv_scheduler = DDIMInverseScheduler.from_config(scheduler.config)
inv_scheduler.set_timesteps(STEPS)

guidance_scale = 7.0
prompt         = ""

tok_kwargs = dict(padding="max_length",
                  max_length=tok.model_max_length,
                  truncation=True,
                  return_tensors="pt")
emb_uncond = clip(tok([""], **tok_kwargs).input_ids).last_hidden_state
emb_cond   = clip(tok([prompt], **tok_kwargs).input_ids).last_hidden_state
use_cfg    = guidance_scale != 1.0 or prompt != ""
goal_tensor = torch.from_numpy(goal_np).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
with torch.no_grad():
    goal_lat = vae.encode(goal_tensor).latent_dist.mean * 0.18215
    latent   = goal_lat.clone()

    for t in inv_scheduler.timesteps:
        eps_uncond = unet(latent, t, encoder_hidden_states=emb_uncond).sample
        if use_cfg:
            eps_cond   = unet(latent, t, encoder_hidden_states=emb_cond).sample
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        else:
            eps = eps_uncond

        latent_prev, _ = inv_scheduler.step(
            model_output=eps,
            timestep=t,
            sample=latent,
            return_dict=False
        )
        latent = latent_prev

    recovered_noise = latent.squeeze().cpu().numpy().astype(np.float32)

goal_lat_noise = recovered_noise.flatten().astype(np.float32)

with open("goal_lat_noise2.bin", "wb") as f:
    f.write(goal_lat_noise.tobytes())

with open("goal_lat_noise2.bin", "rb") as f:
    goal_lat_noise = f.read()
goal_lat_noise = np.frombuffer(goal_lat_noise, dtype=np.float32)

assert goal_lat_noise.shape == (576,)
assert goal_lat_noise.dtype == np.float32
print("Computed goal_lat_noise:", goal_lat_noise.shape) 




import numpy as np
import struct
import torch
#from z3 import *

noise = goal_lat_noise
orig_noise = noise.copy()
print("orig noise:")
print(orig_noise)

# https://github.com/pytorch/pytorch/blob/861945100ec5480a304c09843e4c7a0826941c1d/aten/src/ATen/core/TransformationHelper.h#L84
# print(noise.shape) <- (576,)

def inverse_box_muller(x, y):
    r2 = x*x + y*y
    u1 = np.exp(-0.5 * r2)
    u2 = np.arctan2(y, x) / (2.0 * np.pi)
    if u2 < 0.0:
        u2 += 1.0
    return u1, u2

def box_muller_torch(u1, u2):
    # https://github.com/pytorch/pytorch/blob/861945100ec5480a304c09843e4c7a0826941c1d/aten/src/ATen/native/cpu/DistributionTemplates.h#L139
    radius = np.sqrt(-2. * np.log(u1))
    theta = 2 * np.pi * u2
    return radius * np.cos(theta), radius * np.sin(theta)


# undo the box-muller
for k in range(0, 576, 16):
    #print("operating on chunk", k)
    for i in range(8):
        j1 = k+i
        j2 = j1+8
        #print("operating on pair", j1, j2)
        u1, u2 = inverse_box_muller(noise[j1], noise[j2])
        noise[j1], noise[j2] = 1 - u1, u2#inverse_box_muller(noise[j1], noise[j2])

noise_uniform = noise.copy()

print("rerunning transform in forward pass to check...")
noise_check = noise_uniform.copy()
for k in range(0, 576, 16):
    for i in range(8):
        j1 = k+i
        j2 = j1+8
        noise_check[j1], noise_check[j2] = box_muller_torch(1 - noise_check[j1], noise_check[j2])
print("mse=", ((noise_check-orig_noise)**2).mean())


def uniform_real(v):
    return np.float32(np.uint32(v) & ((1 << 24) - 1)) * (1. / (1 << 24))

for i in range(576):
    noise[i] = noise[i] * (1 << 24)

noise_uint32 = noise.astype(np.uint32)

# pad it to 624
mt19937_outputs = list(noise_uint32) + (624-len(noise_uint32))*[0]
mt19937_outputs = [int(x) for x in mt19937_outputs]

class MT19937:
    w, n = 32, 624
    f = 1812433253
    m, r = 397, 31
    a = 0x9908B0DF
    d, b, c = 0xFFFFFFFF, 0x9D2C5680, 0xEFC60000
    u, s, t, l = 11, 7, 15, 18

def get_bit(x, i):
    return (x & (1 << (MT19937.w - i - 1)))

def reverse_bits(x):
    rev = 0
    for i in range(MT19937.w):
        rev = (rev << 1)
        if(x > 0):
            if (x & 1 == 1):
                rev = (rev ^ 1)
            x = (x >> 1)
    return rev

def inv_left(y, a, b):
    return reverse_bits(inv_right(reverse_bits(y), a, reverse_bits(b)))

def inv_right(y, a, b):
    x = 0
    for i in range(MT19937.w):
        if (i < a):
            x |= get_bit(y, i)
        else:
            x |= (get_bit(y, i) ^ ((get_bit(x, i - a) >> a) & get_bit(b, i)))
    return x

def untemper(y):
    x = y
    x = inv_right(x, MT19937.l, ((1 << MT19937.w) - 1))
    x = inv_left(x, MT19937.t, MT19937.c)
    x = inv_left(x, MT19937.s, MT19937.b)
    x = inv_right(x, MT19937.u, MT19937.d)
    return x

# output the state
mt19937_state = [untemper(x) for x in mt19937_outputs]
print("mt19937 state:")
print(mt19937_state)
#s = my_mt19937.try_recover_seed()
#print("possible seed:", s)


def temper_torch(y):
    y ^= (y >> 11)
    y ^= (y << 7) & 0x9d2c5680
    y ^= (y << 15) & 0xefc60000
    y ^= (y >> 18)
    return y

def do_transform_576(mtstate):
    # temper to get our mt19937 outputs
    outputs = np.array([temper_torch(y) for y in mtstate])
    outputs = outputs[:576]
    # run the uniform real transform
    uniform = np.array([uniform_real(v) for v in outputs])
    # run box-muller
    for k in range(0, 576, 16):
        for i in range(8):
            j1 = k+i
            j2 = j1+8
            uniform[j1], uniform[j2] = box_muller_torch(1 - uniform[j1], uniform[j2])
    return uniform
o = do_transform_576(mt19937_state)

pytorch_state = bytearray(5048) # legacy

struct.pack_into("<Q", pytorch_state, 0, 0) # the_initial_seed
struct.pack_into("<i", pytorch_state, 8, 624) # left
struct.pack_into("<i", pytorch_state, 12, 1) # seeded
struct.pack_into("<Q", pytorch_state, 16, 0) # next
struct.pack_into("<624Q", pytorch_state, 24, *mt19937_state) # state
struct.pack_into("<d", pytorch_state, 5016, 0.0) # normal_x
struct.pack_into("<d", pytorch_state, 5024, 0.0) # normal_y
struct.pack_into("<d", pytorch_state, 5032, 0.0) # normal_rho
struct.pack_into("<i", pytorch_state, 5040, 0) # normal_is_valid
print("pytorch state:")
print(pytorch_state)

state_tensor = torch.tensor(list(pytorch_state), dtype=torch.uint8)
g = torch.Generator()
g.set_state(state_tensor)

# dump the torch state
with open("final2.json", "w") as f:
    f.write(json.dumps({"bytes":list(pytorch_state)}))