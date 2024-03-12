import os
import shutil

def move_images(source_dir):
    sub_dirs = os.listdir(source_dir)
    for dirs in sub_dirs:
        images=os.listdir(os.path.join(source_dir,dirs))
        for image in images:
            shutil.move(os.path.join(source_dir,dirs,image),os.path.join(source_dir,image))
    print(f"Images moved successfully in {source_dir} directory.")

if __name__ == "__main__":
    source_directory = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\val"
    move_images(source_directory)
    source_directory = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\test"
    move_images(source_directory)
    source_directory = r"E:\QuickFox\Signature\signature_dataset\triplet_dataset\train"
    move_images(source_directory)