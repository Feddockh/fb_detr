import os
import argparse
import shutil

IMAGE_EXTS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

def parse_args():
    parser = argparse.ArgumentParser(description="Copy images (and optional labels) based on a .txt list of filenames.")
    parser.add_argument("--list", required=True, help="Path to .txt file with image filenames (with or without extension)")
    parser.add_argument("--src-image-dir", required=True, help="Source directory for images")
    parser.add_argument("--dst-image-dir", required=True, help="Destination directory for matched images")
    parser.add_argument("--src-label-dir", default=None, help="Optional source directory for label .txt files")
    parser.add_argument("--dst-label-dir", default=None, help="Optional destination directory for label .txt files")
    return parser.parse_args()

def find_image_path(name_base, src_dir):
    for ext in IMAGE_EXTS:
        path = os.path.join(src_dir, name_base + ext)
        if os.path.exists(path):
            return path
    return None

def main():
    args = parse_args()
    os.makedirs(args.dst_image_dir, exist_ok=True)
    if args.src_label_dir and args.dst_label_dir:
        os.makedirs(args.dst_label_dir, exist_ok=True)

    with open(args.list, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    copied_images = 0
    copied_labels = 0

    for line in lines:
        # support both full paths and just basenames
        name_base = os.path.splitext(os.path.basename(line))[0]

        # Copy image
        src_img_path = find_image_path(name_base, args.src_image_dir)
        if src_img_path:
            dst_img_path = os.path.join(args.dst_image_dir, os.path.basename(src_img_path))
            shutil.copy2(src_img_path, dst_img_path)
            copied_images += 1
        else:
            print(f"[Warning] Could not find image for {name_base} in {args.src_image_dir}")

        # Optionally copy label
        if args.src_label_dir and args.dst_label_dir:
            src_label_path = os.path.join(args.src_label_dir, name_base + '.txt')
            if os.path.exists(src_label_path):
                dst_label_path = os.path.join(args.dst_label_dir, os.path.basename(src_label_path))
                shutil.copy2(src_label_path, dst_label_path)
                copied_labels += 1
            else:
                print(f"[Warning] Label not found for {name_base} in {args.src_label_dir}")

    print(f"\nCopied {copied_images} images to {args.dst_image_dir}")
    if args.src_label_dir and args.dst_label_dir:
        print(f"Copied {copied_labels} labels to {args.dst_label_dir}")

if __name__ == '__main__':
    main()
