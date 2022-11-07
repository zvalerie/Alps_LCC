import numpy as np

def colormap():
    '''
    create a color map associates all the classes(including background) with label colors.
    
    Returns:
        np.ndarray with dimensions (10, 3)
    '''
    
    colormap = np.asarray(
        [
            [0, 0, 0], 
            [128, 0, 0], 
            [0, 128, 0], 
            [128, 128, 0],  
            [0, 0, 128], 
            [128, 0, 128], 
            [0, 128, 128], 
            [128, 128, 128],  
            [64, 0, 0], 
            [192, 0, 0], 
            [64, 128, 0], 
            [192, 128, 0],  
            [64, 0, 128], 
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0]
        ]
    )
    
    return colormap

def vis_seg_mask(mask):
    colors = colormap()
    height, width = mask.shape
    mask_img = np.zeros((height, width, 3), dtype=np.uint8)
    xv, yv = np.meshgrid(np.arange(0, width), np.arange(0, height))
    
    mask_img[yv, xv, :] = colors[mask]
    return mask_img