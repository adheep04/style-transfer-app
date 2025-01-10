from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image


from app.utils.config import get_config
from app.services.transfer_style import transfer_style
from app.model.model import StyleLearner
from app.utils.img_utils import img_path_to_tensor, save_tensor_as_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message" : "test3"}

@app.get("/health")
async def health_check():
    return {"status" : "healthy"}

@app.get("/transfer")
async def start_transfer():
    print('getting config')
    config = get_config()
    
    model_content = StyleLearner()
    model_style = StyleLearner()
    
    print('models loaded')
    
    content_tensor = img_path_to_tensor(config['content_path'])
    style_tensor = img_path_to_tensor(config['style_path'])
    
    print('tensors generated')
    print(f'content shape : {content_tensor.shape}')
    print(f'style shape : {style_tensor.shape}')
    
    img_generator = transfer_style(
        config = config,
        model_content = model_content,
        model_style = model_style,
        content_tensor = content_tensor,
        style_tensor = style_tensor,
    )
    
    for i, img in enumerate(img_generator):
        
        if i == 2:
            break
        
        save_tensor_as_image(
            img_tensor=img.clone(),
            path=f"/app/output_imgs/trial_{config['trial']}/im_{i}.png"
            )
        
    return('finished!')
    
    