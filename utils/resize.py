from PIL import Image
import cv2
import glob
import os
import shutil


old_image_dir = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\datasets\IVUS\images"
new_image_dir = r"C:\Users\Lenna\Google Drive\Projektarbeit_ResearchProject\datasets\IVUS\images_resized"

if __name__ == "__main__":

    for filepath in glob.iglob(os.path.join(old_image_dir, "*.png")):

        img = cv2.imread(filepath)
        res = cv2.resize(img, dsize=(512,512))
        res_img = Image.fromarray(res)
        filename = os.path.basename(filepath)
        res_img.save(os.path.join(new_image_dir, filename))