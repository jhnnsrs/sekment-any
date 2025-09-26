import torch
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from arkitekt_next import easy, register
from mikro_next.api.schema import Image, from_array_like, PartialRGBViewInput, ColorMap, PartialInstanceMaskViewInput, create_reference_view
import numpy as np
import xarray as xr

# Global variables for model initialization
checkpoint = "sam2.1_hiera_base_plus.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
predictor = None

def get_predictor():
    """Initialize the predictor lazily"""
    global predictor
    if predictor is None:
        model = build_sam2(model_cfg, checkpoint)
        predictor = SAM2AutomaticMaskGenerator(model)
    return predictor

def create_instance_mask(masks, image_shape):
    """
    Create an instance segmentation mask from SAM2 masks.
    
    Args:
        masks: List of mask dictionaries from SAM2
        image_shape: Tuple of (height, width) for the output mask
    
    Returns:
        numpy array: Instance mask where each instance has a unique ID (0 = background)
    """
    height, width = image_shape[:2]
    instance_mask = np.zeros((height, width), dtype=np.int32)
    
    # Sort masks by area (largest first) to prioritize larger segments
    sorted_masks = sorted(masks, key=lambda x: x.get('area', 0), reverse=True)
    
    for instance_id, mask_data in enumerate(sorted_masks, start=1):
        # Get the binary segmentation mask
        binary_mask = mask_data['segmentation']
        
        # Only assign pixels that are not already assigned (background = 0)
        instance_mask[binary_mask & (instance_mask == 0)] = instance_id
    
    return instance_mask

@register
def segment_image(image: Image) -> Image:
    """
    Segments an image using the SAM2 model and returns an instance segmentation mask.

    Args:
        image (Image): The input image to be segmented.

    Returns:
        Image: Instance segmentation mask where each object has a unique ID.
    """
    
    # Convert xarray data to numpy array in HWC format (Height, Width, Channels)
    image_array = image.data.isel(t=0, z=0).transpose("y", "x", "c").values
    
    # Ensure we have RGB channels (3 channels) and convert to uint8
    if image_array.shape[-1] > 3:
        image_array = image_array[:, :, :3]
    
    # Convert to uint8 if not already
    if image_array.dtype != np.uint8:
        # Normalize to 0-255 range if needed
        if image_array.max() <= 1.0:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

    with torch.inference_mode():
        mask_generator = get_predictor()
        masks = mask_generator.generate(image_array)
        print(f"Generated {len(masks)} masks")
        
        # Create instance segmentation mask
        instance_mask = create_instance_mask(masks, image_array.shape)
        print(f"Created instance mask with {len(masks)} instances")
        print(f"Instance mask shape: {instance_mask.shape}")
        print(f"Unique instances: {np.unique(instance_mask)}")
        
        # Create a new Image object with the instance mask
        # Keep the same dimensions as input but replace data with instance mask
        coords = image.data.coords.copy()
        
        # Create xarray DataArray for the instance mask
        # Add a singleton channel dimension and match original coordinates
        mask_data = xr.DataArray(
            instance_mask[np.newaxis, np.newaxis, np.newaxis, :, :],  # Add t, z, c dimensions
            dims=['c', 't', 'z', 'y', 'x']
        )
        
        mask_data = mask_data.astype(np.uint8)
        

        print(mask_data.max())
        print(mask_data.min())
        
        # Create new Image object with the mask data
        segmented_image = from_array_like(
            mask_data,
            name=f"{image.name}_instances" if hasattr(image, 'name') else "instance_mask",
            rgb_views=[PartialRGBViewInput(
                cMax=1,
                cMin=0,
                color_map=ColorMap.RAINBOW,
                base_color=(0,0,0),
                contrastLimitMin=mask_data.min(),
                contrastLimitMax=mask_data.max()
            )],
            instance_mask_views=[PartialInstanceMaskViewInput(
                referenceView=create_reference_view(image)
            )]
        )
        
        return segmented_image