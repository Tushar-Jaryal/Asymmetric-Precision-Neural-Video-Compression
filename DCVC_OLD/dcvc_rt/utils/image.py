def crop_to_shape(img, H, W):
    """
    img: ndarray [H', W']
    returns: ndarray [H, W]
    """
    return img[:H, :W]