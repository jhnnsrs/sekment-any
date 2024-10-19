from arkitekt_next import register
import time
from mikro_next.api.schema import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import torch
from pycocotools import mask as mask_utils
mask = mask_utils.decode(annotation["segmentation"])

print(torch.cuda.is_available())



sam = sam_model_registry["default"](checkpoint="sam_vit_h_4b8939.pth")
mask_generator = SamAutomaticMaskGenerator(sam)

@register
def segment_image(image: Image) -> Image:

    x = image.data.isel(t=0, z=0).transpose(*"yxc").compute().to_numpy()

    x = x[:, :, :3]

    print(x)
    masks = mask_generator.generate(x)
    print(masks)
    return image