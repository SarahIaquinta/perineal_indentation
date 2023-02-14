import indentation.model.comsol.convergence.utils as imu
import matplotlib.pyplot as plt


def get_columns_to_extract(dictionnary):
    keys = dictionnary.keys()
    indices = [k for k in keys]
    return indices

def build_new_dictionnary(filename, dictionnary, header_rows):
    new_dictionnary = {}
    indices = get_columns_to_extract(dictionnary)
    for i in indices:
        label = dictionnary[i]
        data = imu.get_column_i(filename, i, header_rows)
        new_dictionnary[label] = data
    return new_dictionnary

if __name__ == "__main__":
    filename = 'b_SiDispl_coarse_ratio=1_fine_ratio=005.txt'
    dict_headers_silicon_displacement = {0 : "Time [s]", 1 : "Silicon displacement [mm]"}

    header_rows, _ = imu.get_headers(filename)
    new_dictionnary = build_new_dictionnary(filename, dict_headers_silicon_displacement, header_rows)
    print('hello')

    plt.figure()
    labels = new_dictionnary.keys()
    time = new_dictionnary["Time [s]"]
    displ = new_dictionnary["Silicon displacement [mm]"]
    plt.plot(time, displ)
    plt.show()