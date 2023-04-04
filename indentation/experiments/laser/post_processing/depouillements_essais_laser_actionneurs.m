clear all;
close all;

nom_repertoire = 'd:\Documents\Bureautique\Recherche\Collaborations\Sherbrooke\StageElFarid\DonneesTestLaser_03_05_20222\';
nom_generique_fichier = 'replay_115465_2022-5-3';
nom_extension = '.csv';

% % pot vibrant : reglage laser : 3400 Hz
% %liste_fichiers = {'',' (1)',' (2)',' (3)',' (4)',' (5)',' (6)',' (7)',' (8)',' (9)',' (10)',' (11)'};
% liste_fichiers = {'',' (1)',' (2)',' (3)',' (4)',' (5)',' (6)_modif',' (7)_modif',' (8)',' (9)',' (10)',' (11)'};
% nom_fichier_sortie = 'pot_vibrant.mat';
% vec_A = [1 1 1 1 1 1 1 1 1 1 1 1];
% vec_f = [20 50 100 200 300 400 500 600 700 800 900 1000];
% T_max = 3; % s

% piezo : reglage laser : 6000 Hz puis 5000 Hz a partir essai (15)
liste_fichiers = {' (12)',' (13)',' (14)',' (15)',' (16)',' (17)',' (18)',' (19)'};
nom_fichier_sortie = 'piezo.mat';
vec_A = [50 50 50 5 10 5 5 10];
vec_f = [9 50 250 300 300 400 500 500];
T_max = 3; % s

% facteur pour convertir les temps lus dans le fichier (de �s a s)
facteur_normalisation_temps = 1e6; % �s => s

% definition des mots, des caracteres et des codes utilises dans le fichier
mot_fin_lecture_champ = 'End';
caractere_separateur = ',';
numero_canal_temps = 3;

% definition du nom du champ associe au donnees a lire
nom_champ_donnees = 'Profile';

%%%%%%%%%%%%%%%%%%%%%%%%%%%

struct_parametres_lecture = struct('nom_champ_donnees',nom_champ_donnees,'numero_canal_temps',numero_canal_temps,'mot_fin_lecture_champ',mot_fin_lecture_champ,'caractere_separateur',caractere_separateur);

vec_frequence_essai = nan(1,length(liste_fichiers));
vec_amplitude_essai = nan(1,length(liste_fichiers));
vec_amplitude_relative_essai = nan(1,length(liste_fichiers));
liste_donnees = cell(1,length(liste_fichiers));
for n_fichier = 1:length(liste_fichiers)
 disp(['traitement fichier numero ' int2str(n_fichier) ' sur ' int2str(length(liste_fichiers))]);
 nom_fichier = [nom_repertoire nom_generique_fichier liste_fichiers{n_fichier} nom_extension];
 [mat_donnees,vec_temps,liste_nom_champ_en_tete,liste_noms_sous_champ_en_tete,liste_valeurs_sous_champ_en_tete,liste_nom_champ_en_tete_mesure,liste_valeur_champ_en_tete_mesure,liste_nom_champ_mesure] = lecture_donnees_laser_LMI(nom_fichier,struct_parametres_lecture);
% tests pour verifier ce qui est lu dans les differentes en-tetes
%  n_en_tete = 1;
%  liste_nom_champ_en_tete{n_en_tete}
%  nb_sous_champs = length(liste_noms_sous_champ_en_tete{n_en_tete})
%  n_sous_champ = nb_sous_champs;
%  liste_noms_sous_champ_en_tete{n_en_tete}{n_sous_champ}{1}
%  liste_nom_champ_en_tete_mesure
%  liste_valeur_champ_en_tete_mesure{1}
%  liste_nom_champ_mesure{1}
 vec_position_profil = zeros(1,size(mat_donnees,2));
 for i = 1:size(mat_donnees,2)
  vec_position_profil(i) = str2double(liste_nom_champ_mesure{end-size(mat_donnees,2)+i});
 end
 vec_temps = (vec_temps-vec_temps(1))/facteur_normalisation_temps;
 mat_donnees_affich = mat_donnees((vec_temps <= T_max),:);
 vec_temps_affich = vec_temps((vec_temps <= T_max));
 vec_booleen = sum(~isnan(mat_donnees_affich),1);
 i_sel = round ( median(find(vec_booleen == max(vec_booleen))) );
% identification des caracteristiques du signal au point ou on a le plus d'information
 f_ech = 1/median(diff(vec_temps_affich));
 F_hanning = 1/2*(1-cos(2*pi*vec_temps_affich/(max(vec_temps_affich))))';
 signal = mat_donnees_affich(:,i_sel);
 N = length(signal);
% vec_TFD_f = abs(fftshift(fft(signal-mean(signal))));
 vec_TFD_f = abs(fftshift(fft((signal-mean(signal)).*F_hanning)));
 vec_f = (-floor(N/2)+(0:(N-1))).*f_ech/N;
 [ii_pos] = find ( (vec_f > 1*f_ech/N ) );
 [~,ii_max] = max(vec_TFD_f(ii_pos));
 f_signal = vec_f(ii_pos(ii_max));
% figure;plot(vec_f,vec_TFD_f,'-k');hold on;plot(vec_f(ii_pos(ii_max)),vec_TFD_f(ii_pos(ii_max)),'or');grid;
 figure;subplot(1,2,1);imagesc(vec_position_profil,vec_temps,mat_donnees);colorbar;xlabel('x (mm)');ylabel('t (s)');hold on;plot([vec_position_profil(i_sel) vec_position_profil(i_sel)],[min(vec_temps) max(vec_temps)],'-w');axis('ij');axis([min(vec_position_profil) max(vec_position_profil) min(vec_temps) max(vec_temps)]);subplot(1,2,2);plot(vec_temps,mat_donnees(:,i_sel),'r');grid;title(['fichier ' int2str(n_fichier) ', F = ' num2str(round(f_signal)) 'Hz, A = ' num2str(vec_A(n_fichier))]);xlabel('t (s)');ylabel('uz (mm)')
 delta_n_max = round(3*f_ech/f_signal)+1;
 A_z = max(signal((end-delta_n_max+1):end))-min(signal((end-delta_n_max+1):end));
 figure;plot(vec_temps_affich((end-delta_n_max+1):end),signal((end-delta_n_max+1):end),'-r');grid;xlabel('t (s)');ylabel('uz (mm)');title(['fichier ' int2str(n_fichier) ', F = ' num2str(round(f_signal)) 'Hz, Az = ' num2str(A_z) ' mm']);
 liste_donnees{n_fichier} = struct('vec_temps',vec_temps,'mat_donnees',mat_donnees,'f_ech',f_ech,'f_signal',f_signal,'A_z',A_z);
 vec_frequence_essai(n_fichier) = f_signal;
 vec_amplitude_essai(n_fichier) = A_z;
 vec_amplitude_relative_essai(n_fichier) = A_z/vec_A(n_fichier);
end

figure;plot(vec_frequence_essai,vec_amplitude_essai,'-r');grid;xlabel('frequence (Hz)');ylabel('amplitude (mm)');
figure;plot(vec_frequence_essai,vec_amplitude_relative_essai,'-r');grid;xlabel('frequence (Hz)');ylabel('amplitude relative (mm)');

%save([nom_repertoire nom_fichier_sortie])

