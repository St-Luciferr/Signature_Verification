import os
import csv

def create_siamese_csv(dataset_dir, output_csv):
    sub_dirs = os.listdir(dataset_dir)
            
    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["image1", "image2", "label"])

        for dir in sub_dirs:
            images = os.listdir(os.path.join(dataset_dir, dir))
            original_images = [img for img in images if "original" in img]
            for image in original_images:
                for sample in images:
                    if sample != image:
                        if "original" in sample:
                            writer.writerow([image, sample, 1])
                        else:
                            writer.writerow([image, sample, 0])
                    else:
                        continue
if __name__ == "__main__":
    dataset_directory = r"E:\QuickFox\Signature\signature_dataset\split\test"
    output_csv_file = r"E:\QuickFox\Signature\signature_dataset\split\test_data.csv"

    create_siamese_csv(dataset_directory, output_csv_file)
