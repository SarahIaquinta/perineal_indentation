import indentation.model.comsol.convergence as im

filename = 'silicon_displ_coarseratio=1_fineratio=005.txt'

# path_to_file = im.utils.get_path_to_file(filename)

column_i = im.utils.get_column_i(filename, 0)

print(column_i)