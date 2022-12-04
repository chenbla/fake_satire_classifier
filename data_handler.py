from pathlib import Path
import os


def read_file(file_path: Path) -> str:
    with open(str(file_path), 'r', errors='ignore', encoding='utf-8') as f:
        content = f.read()
    return content


def read_all_files(folder_path: Path) -> list:
    # get all the text files in the folder
    txt_full_files_paths = []
    for file in os.listdir(folder_path):
        current_path = os.path.join(folder_path, file)
        if os.path.isfile(current_path):
            if file.endswith(".txt"):
                txt_full_files_paths.append(Path(current_path))

    # read all the files
    content_list = []
    for file_path in txt_full_files_paths:
        content = read_file(file_path)
        content_list.append(content)

    return content_list


def read_data(fake_dataset_dir: Path, satire_dataset_dir) -> dict:
    print(fake_dataset_dir)
    fake_data = read_all_files(fake_dataset_dir)
    satire_data = read_all_files(satire_dataset_dir)
    return {'fake': fake_data, 'satire': satire_data}