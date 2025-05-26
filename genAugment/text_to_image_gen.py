from huggingface_hub import InferenceClient
from tqdm import tqdm
import random
import os
import time
import huggingface_hub
from datetime import datetime

"""
Purpose of script is image generation for two least represented classes of AffectNet: fear and disgust
"""

#unique filename for new images so there won't be any conflicts
def get_unique_filename(dir,prefix='img_',extension='.png'):
    while True:
        filename = f"{prefix}{os.urandom(6).hex()}{extension}"
        file_path = os.path.join(dir,filename)

        if not os.path.exists(file_path):
            return file_path

#set of characteristics that are randomly picked for the prompt
characteristics = {'sex':['male','female'],
                   'age group':['child','teen','young adult','adult','senior'],
                   'ethnicity':['caucasian','african','african-american','asian','indian','middle eastern'],
                   'face shape': ['round','oval','squared','diamond','heart'],
                   'jawline' : ['sharp','soft','square','narrow'],
                   'beard' : ['full beard','goatee','stubble','clean-shaven'],
                   'makeup' : ['full makeup','natural makeup','bold makeup','lipstick','eyeliner','no makeup'],
                   'women hairstyle': ['updo','ponytail','bun','braids'],
                   'hair': ['short','long'],
                   'hair type': ['straight','curly','wavy'],
                   'nose shape' : ['broad','narrow','upturned'],
                   'other' : ['freckles','mole','pimples','glasses','scar',''],
                }

#There are used to simulate real world settings that fit to the images that the end-product will use
styles = [
    "standard webcam feed, soft ambient lighting, slight grain, low resolution, natural room light, casual setting, everyday environment, minimal contrast",
    "low-light webcam capture, slight noise, indoor lighting, fluorescent bulbs, washed out colors, subtle motion blur, typical home lighting, slightly dim",
    "overhead lighting, everyday setting, webcam feel, soft details, muted tones, subtle shadows, minimal contrast, indoor room lighting",
    "low-quality webcam, poor lighting, grainy texture, natural shadows, cool lighting from a computer screen, slightly blurred, everyday indoor environment",
    "grainy webcam feed, artificial lighting, slightly overexposed, visible compression artifacts, casual indoor setting, average resolution, dim background",
    "outdoor webcam capture, slight motion blur, daylight, natural shadows, low resolution, hazy details, uneven lighting, typical security camera quality",
    "midday lighting, webcam-quality image, flat lighting, casual environment, slight grain, soft focus, subtle compression, minimal detail, slightly washed-out colors",
    "overexposed webcam feed, direct sunlight through windows, washed-out highlights, low dynamic range, mild noise, casual background, slightly blurred edges",
    "low-res security camera, grainy footage, harsh lighting, visible pixelation, limited color depth, flat shadows, slight lens distortion, everyday surroundings",
    "grainy surveillance camera, uneven lighting, slightly out of focus, mild distortion, casual background, everyday activity, realistic home camera feed",
    "smartphone front camera, selfie-like quality, natural lighting from windows, soft shadows, slight grain, mid-level detail, mild compression, everyday environment",
    "mid-lighting, security cam view, subtle noise, low sharpness, visible grain, poor resolution, slight blur, typical indoor setup, mild lens distortion",
    "low-light webcam, noise, cool light from computer screen, natural background, shadows, slight motion blur, casual lighting, lower image quality",
    "basic webcam view, standard indoor lighting, soft focus, muted tones, mild motion blur, average detail, subtle pixelation, everyday computer setup",
    "home security cam style, slightly grainy, uneven lighting, minimal contrast, compressed video feed, flat colors, mild noise, casual indoor scene"
]

#part of propt that has to do with facial characteristics
def get_basic_caption(chars: dict):

    #caption = [age group] [ethnicity] [sex] with [hair type] [hair], [nose shape] nose and [other]
    sex = random.choice(chars['sex'])
    age_group = random.choice(chars['age group'])
    ethnicity = random.choice(chars['ethnicity'])
    face_shape = random.choice(chars['face shape'])
    hair = random.choice(chars['hair'])
    hair_type = random.choice(chars['hair type'])
    nose_shape = random.choice(chars['nose shape'])
    other = random.choice(chars['other'])


    if age_group == 'child' or age_group == 'teen':
        caption = f'{age_group} {ethnicity} {sex}, {face_shape} shaped face, {hair_type} {hair} hair, {nose_shape} nose, {other}'

    else: 
        if sex == 'female':
            makeup = random.choice(chars['makeup'])
            hairstyle = random.choice(chars['women hairstyle'])
            caption = f'{age_group} {ethnicity} {sex}, {face_shape} shaped face, {hair_type} {hair} hair, {nose_shape} nose , {makeup}, {hairstyle} hairstyle, {other}'
        else:
            beard = random.choice(chars['beard'])
            jawline = random.choice(chars['jawline'])
            caption = f'{age_group} {ethnicity} {sex}, {face_shape} shaped face, {jawline} jawline, {beard} , {hair_type} {hair} hair, {nose_shape} nose, {other}'
        
    
    return caption
        
token = "HF_TOKEN"

#any other fitting model can be used
# model = "Shakker-Labs/FLUX.1-dev-LoRA-AntiBlur" #cant quite capture disgust everytime
model = "black-forest-labs/FLUX.1-schnell"  #did good ,captured disgust pretty well
# model = "black-forest-labs/FLUX.1-dev" #takes long to response
# model = "Shakker-Labs/FLUX.1-dev-LoRA-add-details" #takes too long
# model = "XLabs-AI/flux-RealismLora" #also good
# model = "prithivMLmods/Canopus-LoRA-Flux-FaceRealism" 
# model = 'VideoAditor/Flux-Lora-Realism'
# model = "strangerzonehf/Flux-Super-Realism-LoRA"
# model = "stabilityai/stable-diffusion-3.5-large-turbo"
# model = "strangerzonehf/Flux-Midjourney-Mix2-LoRA"

huggingface_hub.interpreter_login()
client = InferenceClient(model=model,provider="hf-inference",token=token,headers = {'x-use-cache':'0'})

#negative prompt removes unwanted artifacts
neg_prompt = "shiny skin, disfigured, cartoon, anime, b&w , (wrinkle:1.1), (bad proportions, unnatural feature, incongruous feature:1.4), (blurry, un-sharp, fuzzy, un-detailed skin:1.2),(facial contortion, poorly drawn face, deformed iris, deformed pupils:1.3), (mutated hands and fingers:1.5), disconnected hands, disconnected limbs, skin artifacts"
#part of prompt that pertains to facial expression
fear_expression = 'fear facial expression, worried look , terrified look, raised straight eyebrows, wide open eyes , open mouth, frightened, in shock'
disgust_expression = 'disgusted facial expression, horrible taste, horrible smell, squinted face, scrunched nose, wrinkled forehead, squinted eyes, lifted upper lip'

dst_path = "DESTINATION_DIR_FOR_GENERATED_IMAGES"

for i in tqdm(range(1000),leave = True,unit='image'):

    caption = get_basic_caption(characteristics)
    style = random.choice(styles)
    prompt = f"{fear_expression}, very scared {caption}, facing the camera,close-up photograph, centered, highly detailed face, {style}"

    print("\n",prompt)
    try:
        image = client.text_to_image(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=50,
            height=448,
            width=448,
            guidance_scale=10,
        )

    #this was added because of the usage limit of the API, generation resumes as soon as usage limit is reset
    except huggingface_hub.utils.HfHubHTTPError or KeyboardInterrupt as e :
        print(e)
        if e.response.status_code == 429:
            now = datetime.now()
            seconds_needed = (60-now.minute)*60 - now.second
            print(f"\nToo many requests, need to wait for {seconds_needed} sec.")
            time.sleep(seconds_needed)
        continue

    img_path = get_unique_filename(dst_path)
    image.save(img_path,format='PNG')