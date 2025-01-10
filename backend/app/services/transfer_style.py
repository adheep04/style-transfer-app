import torch.nn as nn
import torch
from torch.optim import lr_scheduler


# global constants
device = torch.device('cuda')

# given learnable image parameters and the content repr of the original
# returns the prediction F image content
def pred(model, img_params):
    
    # shallow copy of learnable image ensures gradients aren't affected
    img_tensor = img_params.clone().float().to(device)
    assert (1, 3, 224, 224) == img_tensor.shape, f"Shape mismatch: {img_tensor.shape} should be (1, 3, 224, 224)"
    
    return model(img_tensor)

# the following loss class is not vectorized and may seem inefficient, however
# designing it in the following way helped me understand it
class StyleContentLoss(nn.Module):
    def __init__(self,):
        super().__init__()
        
        # using mean-squared error loss for content loss
        self.mse = nn.MSELoss()
        
        # n as layers progress
        self.N = [64, 128, 256, 512, 512, 512]
        
        # m as layers progress
        self.M = [224, 112, 56, 28, 14, 7]
        
    # computes total loss
    def forward(self, pred_content, pred_styles, true_content, true_styles, alpha, beta, style_weights):
        
        # calculate content_loss and multiply by its weight
        content_loss =  alpha * self.mse(pred_content, true_content)
        
        # calculate style_loss and multiply by its weight
        style_loss = beta * self.total_style_loss(pred_styles, true_styles, style_weights)
        
        return (content_loss + style_loss), content_loss, style_loss
        
    # returns the style layer loss given 1) pred x 2) label a, 3) current layer number
    def layer_style_loss(self, x, a, i):
        
        # general error calculation
        style_error = torch.sum((x - a) ** 2)
        
        # normalize error using n and m
        loss_normalization = (4*(self.N[i]**2)*(self.M[i]**2))
        return style_error/loss_normalization
    
    def total_style_loss(self, x_list, a_list, w):
        
        # convert config weight list to tensor
        w = torch.tensor(w)
        
        # calculate sum of the style losses per layer
        style_loss = 0
        for i in range(len(x_list)):
            style_loss += self.layer_style_loss(x_list[i], a_list[i], i)
        
        return style_loss

# content_image, style_image -> stylized content 
def transfer_style(
    config,
    model_content,
    model_style,  
    content_tensor,
    style_tensor,
):
    
    loss_fn = StyleContentLoss()
    content_layer = config['content_layer']
    style_layer = config['style_layer']
        
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # generate blank brown image as a tensor
    pred_params = torch.rand(1, 3, 224, 224, device=device) 
    
    # Make it learnable
    pred_params = torch.nn.Parameter(pred_params)  
    
    # process and move img_content and img_style to GPU
    content_tensor = content_tensor.to(device)
    style_tensor = style_tensor.to(device)
    
    # initialize models
    model_content = model_content.to(device) # to extract photograph content
    model_style = model_style.to(device) # to extract painting style
    
    # getting content representation from content image from given layer in model
    output_c = model_content(content_tensor)
    true_content = output_c[0][content_layer].detach() # detaching so it won't be included in gradient descent update
    
    # getting style representations from style image UP TO AND INCLUDING GIVEN LAYER (for total style loss calculation)
    output_s = model_style(style_tensor)
    
    # detach each layer's style representation in the list individually using list comprehension
    true_styles = [style.detach() for style in output_s[1][0:style_layer+1]]
    
    # initialize optimizer
    optimizer = torch.optim.Adam([pred_params], lr=config['lr'])
    
    # learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], patience=config['patience'])
    
    for step in range(config['max_steps']):
        # reset gradients
        optimizer.zero_grad()

        # content prediction
        pred_content = pred(model_content, pred_params)[0][content_layer]

        # style prediction
        pred_styles = pred(model_style, pred_params)[1][0:style_layer+1]

        # calculate loss
        alpha = config['alpha']
        beta = config['beta']
        style_weights = config['wl']
        loss, content_loss, style_loss = loss_fn(pred_content, pred_styles, true_content, true_styles, alpha, beta, style_weights)
            
        # backpropagate and calculate gradients
        loss.backward()

        # update scheduler
        scheduler.step(loss.item())
        optimizer.step()
    
        # save image every 20 steps
        if step%config['freq'] == 0:
            yield {
            'tensor' : pred_params.clone().detach(),
            'style_loss' : style_loss.item(),
            'content_loss' : content_loss.item(),
            'step' : step,
            'norm' : torch.norm(pred_params.grad),
            'loss' : (style_loss.item() + content_loss.item())
            }
        
    return pred_params.float()

