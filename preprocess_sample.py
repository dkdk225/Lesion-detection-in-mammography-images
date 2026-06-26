import pydicom
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from math import isnan

df = pd.read_csv('./VinDr_Mammo_Labels/finding_annotations.csv')


def convert_dicom_to_png(dicom_file: str) -> np.ndarray:
    """
    dicom_file: path to the dicom fife

    return
        gray scale image with pixel intensity in the range [0,255]
        None if cannot convert

    """
    data = pydicom.read_file(dicom_file)
    if ('WindowCenter' not in data) or\
       ('WindowWidth' not in data) or\
       ('PhotometricInterpretation' not in data) or\
       ('RescaleSlope' not in data) or\
       ('PresentationIntentType' not in data) or\
       ('RescaleIntercept' not in data):

        print(f"{dicom_file} DICOM file does not have required fields")
        return

    intentType = data.data_element('PresentationIntentType').value
    if ( str(intentType).split(' ')[-1]=='PROCESSING' ):
        print(f"{dicom_file} got processing file")
        return


    c = data.data_element('WindowCenter').value # data[0x0028, 0x1050].value
    w = data.data_element('WindowWidth').value  # data[0x0028, 0x1051].value
    if type(c)==pydicom.multival.MultiValue:
        c = c[0]
        w = w[0]

    photometricInterpretation = data.data_element('PhotometricInterpretation').value

    try:
        a = data.pixel_array
    except:
        print(f'{dicom_file} Cannot get get pixel_array!')
        return

    slope = data.data_element('RescaleSlope').value
    intercept = data.data_element('RescaleIntercept').value
    a = a * slope + intercept

    try:
        pad_val = data.get('PixelPaddingValue')
        pad_limit = data.get('PixelPaddingRangeLimit', -99999)
        if pad_limit == -99999:
            mask_pad = (a==pad_val)
        else:
            if str(photometricInterpretation) == 'MONOCHROME2':
                mask_pad = (a >= pad_val) & (a <= pad_limit)
            else:
                mask_pad = (a >= pad_limit) & (a <= pad_val)
    except:
        # Manually create padding mask
        # this is based on the assumption that padding values take majority of the histogram
        print(f'{dicom_file} has no PixelPaddingValue')
        a = a.astype(np.int)
        pixels, pixel_counts = np.unique(a, return_counts=True)
        sorted_idxs = np.argsort(pixel_counts)[::-1]
        sorted_pixel_counts = pixel_counts[sorted_idxs]
        sorted_pixels = pixels[sorted_idxs]
        mask_pad = a == sorted_pixels[0]
        try:
            # if the second most frequent value (if any) is significantly more frequent than the third then
            # it is also considered padding value
            if sorted_pixel_counts[1] > sorted_pixel_counts[2] * 10:
                mask_pad = np.logical_or(mask_pad, a == sorted_pixels[1])
                print(f'{dicom_file} most frequent pixel values: {sorted_pixels[0]}; {sorted_pixels[1]}')
        except:
            print(f'{dicom_file} most frequent pixel value {sorted_pixels[0]}')

    # apply window
    mm = c - 0.5 - (w-1)/2
    MM = c - 0.5 + (w-1)/2
    a[a<mm] = 0
    a[a>MM] = 255
    mask = (a>=mm) & (a<=MM)
    a[mask] = ((a[mask] - (c - 0.5)) / (w-1) + 0.5) * 255

    if str( photometricInterpretation ) == 'MONOCHROME1':
        a = 255 - a

    a[mask_pad] = 0
    return a

def convert_dicom_to_png_in_batch(path_list: list, output_folder: str):
    """
    path_list: list of paths to the dicom files
    output_folder: path to the output folder

    """
    for path in path_list:
        png_img = convert_dicom_to_png(path)
        name = path.split('/')[-1].replace('.dicom', '.png')
        if png_img is not None:
            cv2.imwrite(f"{output_folder}/{name}", png_img)

#converts images to png and crops them then generates labels accordingly
def preprocess(data_frame: pd.DataFrame,image_output_folder: str, label_output_folder: str, path_to_images: str):
    annotation_dict = {}
    for idx, row in data_frame.iterrows():
        path = f"{path_to_images}/{row['study_id']}/{row['image_id']}.dicom"
        png_img = convert_dicom_to_png(path)
        #crop the black parts of the image
        gray = np.clip(png_img, 0, 255).astype('uint8')
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_mammogram = png_img[y:y+h, x:x+w]

        #TODO GET ALL LESIONS
        #  recalc the xmin, xmax, ymin, ymax according to the cropped image

        cases = df[df["image_id"] == row['image_id']]
        label = []
        for _,case in cases.iterrows():
            if not isnan(case['xmin']):
                def clamp(val, min_val, max_val):
                    return max(min(val, max_val), min_val)
                xmin = clamp(case['xmin']-x, 0, w)
                xmax = clamp(case['xmax']-x, 0, w)
                ymin = clamp(case['ymin']-y, 0, h)
                ymax = clamp(case['ymax']-y, 0, h)
                xc = (xmin + xmax) / 2 / w
                yc = (ymin + ymax) / 2 / h
                width  = (xmax - xmin) / w
                height  = (ymax - ymin) / h
                label.append(f"0 {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}")
        

        #create a dict to temporarily store the labels for each image
        if row['image_id'] not in annotation_dict:
            annotation_dict[row['image_id']] = []
        annotation_dict[row['image_id']].append(label)

        #save the cropped image to the output folder
        if png_img is not None:
            cv2.imwrite(f"{image_output_folder}/{row['image_id']}.png", cropped_mammogram)

        with open(f"{label_output_folder}/{row['image_id']}.txt", "w") as f:
            f.write("\n".join(label))



paths = ["preprocessed_dataset/images/train", "preprocessed_dataset/images/val", "preprocessed_dataset/labels/train", "preprocessed_dataset/labels/val"]

print("Reading finding_annotations.csv...")
malignant = pd.read_csv('./dataset/labels/train/malignant.csv').sample(frac=0.8, random_state=42)
benign = pd.read_csv('./dataset/labels/train/benign.csv').sample(frac=0.8, random_state=42)
malignant_val = pd.read_csv('./dataset/labels/train/malignant.csv').drop(malignant.index)
benign_val = pd.read_csv('./dataset/labels/train/benign.csv').drop(benign.index)
malignant_test = pd.read_csv('./dataset/labels/val/malignant.csv')
benign_test = pd.read_csv('./dataset/labels/val/benign.csv')

print("Creating folders...")
for path in paths: Path(path).mkdir(parents=True, exist_ok=True)
print("Converting dicom images to png and creating labels...")
 
preprocess(malignant, paths[0], paths[2], "./dataset/images/train/malignant")
preprocess(benign, paths[0], paths[2], "./dataset/images/train/benign")
preprocess(malignant_val, paths[1], paths[3], "./dataset/images/train/malignant")
preprocess(benign_val, paths[1], paths[3], "./dataset/images/train/benign")
preprocess(malignant_test, paths[1], paths[3], "./dataset/images/test/malignant")
preprocess(benign_test, paths[1], paths[3], "./dataset/images/test/benign")

print("Done.")