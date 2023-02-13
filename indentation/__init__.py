from path import Path

root_path = Path(__file__).abspath().parent


def make_data_folder(folder_name):
    path_to_data_folder = root_path / folder_name
    Path.mkdir_p(path_to_data_folder)
    return path_to_data_folder

