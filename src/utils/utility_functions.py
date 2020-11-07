import os


def binarize_images(images_array, threshold=0.2):
    return (images_array > threshold) * 1.0


def check_exist_folder(folder_path, create_if_not_exist=True):
    is_exist = os.path.exists(folder_path)
    if not is_exist and create_if_not_exist:
        os.makedirs(folder_path)
        return create_if_not_exist
    return is_exist
