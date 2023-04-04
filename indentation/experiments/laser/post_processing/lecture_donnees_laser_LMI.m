function [mat_donnees,vec_temps,liste_nom_champ_en_tete,liste_noms_sous_champ_en_tete,liste_valeurs_sous_champ_en_tete,liste_nom_champ_en_tete_mesure,liste_valeur_champ_en_tete_mesure,liste_nom_champ_mesure] = lecture_donnees_laser_LMI(nom_fichier,struct_parametres_lecture)

nom_champ_donnees = struct_parametres_lecture.nom_champ_donnees;
nom_fin_lecture_champ = struct_parametres_lecture.mot_fin_lecture_champ;
char_separateur = struct_parametres_lecture.caractere_separateur;
n_canal_temps = struct_parametres_lecture.numero_canal_temps;

% on lit toutes les lignes du fichier pour connaitre le nombre de données d'en-tete et le nombre de données 
fid = fopen(nom_fichier,'r');
test = false;
while ( ~test )
 a = fgetl(fid);
 if ( strcmp(a,nom_champ_donnees) == 1 )
  test = true;
 end
end
a = fgetl(fid);
a = fgetl(fid);
a = fgetl(fid);
n_ligne = 0;
test = false;
while ( ~test )
 a = fgetl(fid);
 if ( ~ (strcmp(a,nom_fin_lecture_champ) == 1) )
  n_ligne = n_ligne+1;
 else
  test = true;
 end
end
fclose(fid);

% on lit le fichier
nb_lignes_donnees = n_ligne;
fid = fopen(nom_fichier,'r');
test_lecture_champ = false;
test_lecture_donnees = false;
liste_nom_champ_en_tete = {};
liste_noms_sous_champ_en_tete = {};
liste_valeurs_sous_champ_en_tete = {};
while ( ~feof(fid) )
 a = fgetl(fid);
 if ( ~isempty(a) )
  if ( ~test_lecture_donnees )
   nom_champ = a;
   if ( ~(strcmp(nom_champ,nom_champ_donnees) == 1) )
    liste_nom_champ_en_tete{length(liste_nom_champ_en_tete)+1} = nom_champ;
    test_local = false;
    liste_noms_sous_champs = {};
    liste_valeurs_sous_champs = {};
    while ( ~test_local )
     a_1 = fgetl(fid);
     a_2 = fgetl(fid);
     a_3 = fgetl(fid);
     ii_separateur_1 = strfind(a_1,char_separateur);
     liste_nom_tempo = cell(1,length(ii_separateur_1)+1);
     i_deb = 1;
     for i = 1:length(ii_separateur_1)
      i_fin = ii_separateur_1(i)-1;
      liste_nom_tempo{i} = a_1(i_deb:i_fin);
      i_deb = ii_separateur_1(i)+1;
     end
     i_fin = length(a_1);
     liste_nom_tempo{end} = a_1(i_deb:i_fin);
     ii_separateur_2 = strfind(a_2,char_separateur);
     liste_valeur_tempo = cell(1,length(ii_separateur_2)+1);
     i_deb = 1;
     for i = 1:length(ii_separateur_2)
      i_fin = ii_separateur_2(i)-1;
      liste_valeur_tempo{i} = a_2(i_deb:i_fin);
      i_deb = ii_separateur_2(i)+1;
     end
     i_fin = length(a_2);
     liste_valeur_tempo{end} = a_2(i_deb:i_fin);
     liste_noms_sous_champs{length(liste_noms_sous_champs)+1} = liste_nom_tempo;
     liste_valeurs_sous_champs{length(liste_valeurs_sous_champs)+1} = liste_valeur_tempo;
     if ( strcmp(a_3,nom_fin_lecture_champ) == 1 )
      test_local = true;
     end
    end
    liste_noms_sous_champ_en_tete{length(liste_noms_sous_champ_en_tete)+1} = liste_noms_sous_champs;
    liste_valeurs_sous_champ_en_tete{length(liste_valeurs_sous_champ_en_tete)+1} = liste_valeurs_sous_champs;
   elseif ( strcmp(nom_champ,nom_champ_donnees) == 1 )
    test_lecture_donnees = true;
    a_1 = fgetl(fid);
    a_2 = fgetl(fid);
    a_3 = fgetl(fid);
    ii_separateur_1 = strfind(a_1,char_separateur);
    liste_nom_champ_en_tete_mesure = cell(1,length(ii_separateur_1)+1);
    i_deb = 1;
    for i = 1:length(ii_separateur_1)
     i_fin = ii_separateur_1(i)-1;
     liste_nom_champ_en_tete_mesure{i} = a_1(i_deb:i_fin);
     i_deb = ii_separateur_1(i)+1;
    end
    i_fin = length(a_1);
    liste_nom_champ_en_tete_mesure{end} = a_1(i_deb:i_fin);
    ii_separateur_2 = strfind(a_2,char_separateur);
    liste_valeur_champ_en_tete_mesure = cell(1,length(ii_separateur_2)+1);
    i_deb = 1;
    for i = 1:length(ii_separateur_2)
     i_fin = ii_separateur_2(i)-1;
     liste_valeur_champ_en_tete_mesure{i} = a_2(i_deb:i_fin);
     i_deb = ii_separateur_2(i)+1;
    end
    i_fin = length(a_2);
    liste_valeur_champ_en_tete_mesure{end} = a_2(i_deb:i_fin);
    ii_separateur_3 = strfind(a_3,char_separateur);
    liste_nom_champ_mesure = cell(1,length(ii_separateur_3)+1);
    i_deb = 1;
    for i = 1:length(ii_separateur_3)
     i_fin = ii_separateur_3(i)-1;
     liste_nom_champ_mesure{i} = a_3(i_deb:i_fin);
     i_deb = ii_separateur_3(i)+1;
    end
    i_fin = length(a_3);
    liste_nom_champ_mesure{end} = a_3(i_deb:i_fin);
%    nb_mesures = str2double(liste_valeur_champ_en_tete_mesure{1});
    nb_donnees_mesurees = length(liste_nom_champ_mesure);
    n_debut_profil = 1;
    while ( isnan(str2double(liste_nom_champ_mesure{n_debut_profil})) )
      n_debut_profil = n_debut_profil+1;
    end
    vec_temps = nan(1,nb_lignes_donnees);
    mat_donnees = nan(nb_lignes_donnees,nb_donnees_mesurees-n_debut_profil+1);
    n_ligne_donnees = 0;
   end
  elseif ( test_lecture_donnees )
   if ( ~(strcmp(a,nom_fin_lecture_champ) == 1) )
    n_ligne_donnees = n_ligne_donnees+1;
    test = false;
    while ( ~test )
     ii = strfind(a,[char_separateur char_separateur]);
     if ( isempty(ii))
      test = true;
     else
      a_new = [a(1:(ii(1)-1)) char_separateur ' ' char_separateur a((ii(1)+2:end))];
      a = a_new;
     end
    end
    [ii_separateur] = strfind (a,char_separateur);
    val_temps = str2double(a((ii_separateur(n_canal_temps-1)+1):(ii_separateur(n_canal_temps)-1)));
    vec_donnees = nan(1,size(mat_donnees,2));
    for n_donnees = 1:size(mat_donnees,2)-1
     val_donnee = str2double(a((ii_separateur((n_debut_profil+n_donnees-1)-1)+1):(ii_separateur((n_debut_profil+n_donnees-1))-1)));
     vec_donnees(n_donnees) = val_donnee;
    end
    val_donnee = str2double(a((ii_separateur((n_debut_profil+size(mat_donnees,2)-1-1)-1)+1):end));
    vec_donnees(end) = val_donnee;
    vec_temps(n_ligne_donnees) = val_temps;
    mat_donnees(n_ligne_donnees,:) = vec_donnees;
   end
  end
 end
end
fclose(fid);

