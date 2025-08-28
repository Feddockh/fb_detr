import os
import argparse

#!/usr/bin/env python3


def list_all_files(directory):
    file_list = []
    for root, _, files in os.walk(directory):
        for filename in files:
            file_list.append(os.path.join(root, filename))
    return file_list

def save_filenames(filenames, output_file):
    with open(output_file, 'w') as f:
        for filename in filenames:
            f.write(f"{filename}\n")

def main():
    parser = argparse.ArgumentParser(
        description="List all filenames in a folder (recursively) and save them to a text file."
    )
    parser.add_argument("folder", help="The folder to search for files.")
    parser.add_argument(
        "--output", 
        default="filenames.txt", 
        help="Output file to write the filenames (default: filenames.txt)."
    )
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Error: '{args.folder}' is not a valid directory.")
        exit(1)

    filenames = list_all_files(args.folder)
    save_filenames(filenames, args.output)
    print(f"Saved {len(filenames)} filenames to '{args.output}'.")

if __name__ == "__main__":
    main()