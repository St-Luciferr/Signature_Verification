import os
import csv
from itertools import combinations

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

def create_pairs_csv(dataset_dir, output_csv):
    sub_dirs = os.listdir(dataset_dir)
    # classes = [folder for folder in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, folder))]
    # pairs = []

    with open(output_csv, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["anchor", "positive", "negative"])
        for dir in sub_dirs:
            class_folder = os.path.join(dataset_dir, dir)
            images = os.listdir(class_folder)
            original_images = [img for img in images if "original" in img]
            forged_images = [img for img in images if "forgeries" in img]
            for image in original_images:
                    for positive,negative  in zip(original_images,forged_images):
                        if positive != image:
                            writer.writerow([image, positive, negative])
                        else:
                            continue

    print(f"CSV file '{output_csv}' created successfully.")


if __name__ == "__main__":
    dataset_directory = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\val"
    output_csv_file = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\val_data.csv"
    create_pairs_csv(dataset_directory, output_csv_file)

    dataset_directory = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\test"
    output_csv_file = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\test_data.csv"
    create_pairs_csv(dataset_directory, output_csv_file)

    dataset_directory = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\train"
    output_csv_file = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\train_data.csv"
    create_pairs_csv(dataset_directory, output_csv_file)