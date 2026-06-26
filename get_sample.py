import pandas as pd
from pathlib import Path

print("Reading finding_annotations.csv...")
df = pd.read_csv('./VinDr_Mammo_Labels/finding_annotations.csv')
print("Creating samples...")

#Training set: 900 malignant, 300 benign
malignant = df[(df["finding_categories"] != "['No Finding']") & (df["split"] == "training")]
malignant_unique = malignant.drop_duplicates(subset="image_id")
malignant_unique_900 = malignant_unique.sample(n=900,random_state=42)
malignant_unique_downloads = [f"https://physionet.org/files/vindr-mammo/1.0.0/images/{x['study_id']}/{x['image_id']}.dicom?download \n  dir=./dataset2/images/train/malignant/{x['study_id']} \n  out={x['image_id']}.dicom" for x in malignant_unique_900.iloc]
malignant_unique_downloads = " \n\n".join(malignant_unique_downloads)

benign = df[(df["finding_categories"] == "['No Finding']") & (df["split"] == "training")]
benign_unique = benign.drop_duplicates(subset="image_id")
benign_unique_300 = benign_unique.sample(n=300,random_state=42)
benign_unique_downloads = [f"https://physionet.org/files/vindr-mammo/1.0.0/images/{x['study_id']}/{x['image_id']}.dicom?download \n  dir=./dataset2/images/train/benign/{x['study_id']} \n  out={x['image_id']}.dicom" for x in benign_unique_300.iloc]
benign_unique_downloads = " \n\n".join(benign_unique_downloads)


#Testing set: 30 malignant, 10 benign
malignant_test = df[(df["finding_categories"] != "['No Finding']") & (df["split"] == "test")].sample(n=30, random_state=42)
benign_test = df[(df["finding_categories"] == "['No Finding']") & (df["split"] == "test")].sample(n=10, random_state=42)

benign_test_downloads = [f"https://physionet.org/files/vindr-mammo/1.0.0/images/{x['study_id']}/{x['image_id']}.dicom?download \n  dir=./dataset/images/test/benign/{x['study_id']} \n  out={x['image_id']}.dicom" for x in benign_test.iloc]
benign_test_downloads = " \n\n".join(benign_test_downloads)

malignant_test_downloads = [f"https://physionet.org/files/vindr-mammo/1.0.0/images/{x['study_id']}/{x['image_id']}.dicom?download \n  dir=./dataset/images/test/malignant/{x['study_id']} \n  out={x['image_id']}.dicom" for x in malignant_test.iloc]
malignant_test_downloads = " \n\n".join(malignant_test_downloads)

#create download files for training and testing sets
#command for batch downloading the files using aria2c
#aria2c -j 6 -x 8 -s 8 -c -i <filename>.txt --load-cookies=cookies.txt
#GET YOUR COOKIE WITH A COOKIE EXPORTER EXTENSION IN NETSCAPE FORMAT
print("Creating download files...")
with open("benign.txt", "w") as f:
    f.write(benign_unique_downloads)  

with open("malignant.txt", "w") as f:
    f.write(malignant_unique_downloads) 

with open("benign_test.txt", "w") as f:
    f.write(benign_test_downloads)  

with open("malignant_test.txt", "w") as f:
    f.write(malignant_test_downloads) 

print("Creating labels for training and testing sets...")
#create labels for training and testing sets
Path("dataset/labels/train").mkdir(parents=True, exist_ok=True)
Path("dataset/labels/val").mkdir(parents=True, exist_ok=True)

malignant_unique_900.to_csv("./dataset/labels/train/malignant.csv", index=False)
benign_unique_300.to_csv("./dataset/labels/train/benign.csv", index=False)
malignant_test.to_csv("./dataset/labels/val/malignant.csv", index=False)
benign_test.to_csv("./dataset/labels/val/benign.csv", index=False)


print("Done")