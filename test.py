# import numpy as np
# import cv2


# def feather_image(image_path, mask_path, feather_amount=10):
#     image = cv2.imread(image_path)
#     mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     if image is None or mask is None:
#         raise ValueError("err")
#     mask = mask / 255.0
#     feathered_mask = cv2.GaussianBlur(mask, (feather_amount*2+1, feather_amount*2+1), 0)
#     feathered_mask = feathered_mask[:, :, np.newaxis]
#     background = np.zeros_like(image, dtype=image.dtype)
#     feathered_image = image * feathered_mask + background * (1 - feathered_mask)
#     return feathered_image

    

# cv2.imwrite('test.jpg', feather_image('./img.jpg', './mask.jpg'))