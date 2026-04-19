from skimage.feature  import hog

def extract_hog_features(image):
    return hog(image,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            block_norm='L2-Hys',
            feature_vector=True
            )