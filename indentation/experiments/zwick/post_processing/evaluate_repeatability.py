"""
The objective of this file is to compare the "recovery" profile when obtained
with the laser and when artificially reproduced with a force-controlled
indenter removal via the zwick pull test machine.
"""

from indentation.experiments.zwick.figures.utils import CreateFigure, Fonts, SaveFigure
import indentation.experiments.zwick.post_processing.identify_force_controlled_recovery as z_id_recovery
from indentation.experiments.zwick.post_processing.read_file import Files_Zwick

def find_ids_with_same_imposed_displacement_zwick(files_zwick, datafile, imposed_displacement):
    metadatafile = files_zwick.import_metadatafile(datafile[0:6])
    imposed_disp_dict, _ = files_zwick.read_metadatas_zwick(metadatafile)
    ids_where_disp_is_imposed_disp_zwick = [id for id, disp in imposed_disp_dict.items() if disp == int(imposed_displacement)]
    return ids_where_disp_is_imposed_disp_zwick

def plot_recovery_data_for_imposed_displacement_zwick(files_zwick, datafile, imposed_displacement, createfigure, savefigure, fonts):
    metadatafile = files_zwick.import_metadatafile(datafile[0:6])
    ids_where_disp_is_imposed_disp_zwick = find_ids_with_same_imposed_displacement_zwick(files_zwick, datafile, imposed_displacement)
    sheets_list = ids_where_disp_is_imposed_disp_zwick
    fig_force_vs_time_during_recovery = createfigure.rectangle_rz_figure(pixels=180)
    fig_disp_vs_time_during_recovery = createfigure.rectangle_rz_figure(pixels=180)
    fig_disp_vs_time_during_recovery_laser_format = createfigure.rectangle_rz_figure(pixels=180)
    fig_force_vs_disp_during_recovery = createfigure.rectangle_rz_figure(pixels=180)
    ax_force_vs_time_during_recovery = fig_force_vs_time_during_recovery.gca()
    ax_disp_vs_time_during_recovery = fig_disp_vs_time_during_recovery.gca()
    ax_disp_vs_time_during_recovery_laser_format = fig_disp_vs_time_during_recovery_laser_format.gca()
    ax_force_vs_disp_during_recovery = fig_force_vs_disp_during_recovery.gca()
    imposed_disp_dict, speed_dict = files_zwick.read_metadatas_zwick(metadatafile)
    marker_dict = {100: 'o', 75: 's', 50: 'D'}
    for sheet in sheets_list:
        force_during_recovery_correct_pid, time_during_recovery_correct_pid, disp_during_recovery_correct_pid = z_id_recovery.remove_error_pid_from_data(files_zwick, datafile, sheet)
        imposed_speed = speed_dict[sheet]
        imposed_disp = imposed_disp_dict[sheet]
        ax_force_vs_time_during_recovery.plot(time_during_recovery_correct_pid, force_during_recovery_correct_pid, marker = marker_dict[imposed_speed], lw=0, markersize=2, label= sheet + ' ; ' + str(imposed_disp) + ' mm ; ' + str(imposed_speed) + 'mm/min')
        ax_disp_vs_time_during_recovery.plot(time_during_recovery_correct_pid, disp_during_recovery_correct_pid, marker = marker_dict[imposed_speed], lw=0, markersize=2, label= sheet + ' ; ' + str(imposed_disp) + ' mm ; ' + str(imposed_speed) + 'mm/min')
        ax_disp_vs_time_during_recovery_laser_format.plot(time_during_recovery_correct_pid, max(disp_during_recovery_correct_pid)-disp_during_recovery_correct_pid, marker = marker_dict[imposed_speed], lw=0, markersize=2, label= sheet + ' ; ' + str(imposed_disp) + ' mm ; ' + str(imposed_speed) + 'mm/min')
        ax_force_vs_disp_during_recovery.plot(force_during_recovery_correct_pid, disp_during_recovery_correct_pid, marker = marker_dict[imposed_speed], lw=0, markersize=2, label= sheet + ' ; ' + str(imposed_disp) + ' mm ; ' + str(imposed_speed) + 'mm/min')
    ax_force_vs_time_during_recovery.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time_during_recovery.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time_during_recovery_laser_format.set_xlabel(r"time [s]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp_during_recovery.set_xlabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time_during_recovery.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time_during_recovery.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_disp_vs_time_during_recovery_laser_format.set_ylabel(r"U [mm]", font=fonts.serif(), fontsize=26)
    ax_force_vs_disp_during_recovery.set_ylabel(r"Force [N]", font=fonts.serif(), fontsize=26)
    ax_force_vs_time_during_recovery.legend(prop=fonts.serif_rz_legend(), loc='center right', framealpha=0.7)
    ax_disp_vs_time_during_recovery.legend(prop=fonts.serif_rz_legend(), loc='center right', framealpha=0.7)
    ax_disp_vs_time_during_recovery_laser_format.legend(prop=fonts.serif_rz_legend(), loc='center right', framealpha=0.7)
    ax_force_vs_disp_during_recovery.legend(prop=fonts.serif_rz_legend(), loc='center right', framealpha=0.7)
    savefigure.save_as_png(fig_force_vs_time_during_recovery, sheet + "_force_vs_time_during_recovery_imposed_disp_" + str(imposed_displacement))
    savefigure.save_as_png(fig_disp_vs_time_during_recovery, sheet + "_disp_vs_time_during_recovery_imposed_disp_" + str(imposed_displacement))
    savefigure.save_as_png(fig_disp_vs_time_during_recovery_laser_format, sheet + "_disp_vs_time_during_recovery_laser_format_imposed_disp_" + str(imposed_displacement))
    savefigure.save_as_png(fig_force_vs_disp_during_recovery, sheet + "_force_vs_disp_during_recovery_imposed_disp_" + str(imposed_displacement))    




if __name__ == "__main__":
    createfigure = CreateFigure()
    fonts = Fonts()
    savefigure = SaveFigure()
    experiment_dates = ['230515']#, '230411']#'230331', '230327', '230403']
    types_of_essay = ['C_Indentation_relaxation_maintienFnulle_500N_trav.xls']#, 'RDG']
    files_zwick = Files_Zwick(types_of_essay[0])
    datafile_list = files_zwick.import_files(experiment_dates[0])
    datafile = datafile_list[0]
    # datafile_as_pds, sheets_list_with_data = files_zwick.get_sheets_from_datafile(datafile)
    imposed_displacement = 5
    plot_recovery_data_for_imposed_displacement_zwick(files_zwick, datafile, imposed_displacement, createfigure, savefigure, fonts)
    # identify_beginning_recovery(files_zwick, datafile, sheet1)

        
    # error_pid_indices = remove_error_pid_from_data(files_zwick, datafile, sheet2)
    print('hello')
    
    
