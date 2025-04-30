import os
import numpy as np
import matplotlib.pyplot as plt
from neuroscience_tools import neuro_data_utils as npu


# === Fonction pour charger et calculer performance ===
def get_performances_from_folder(folder_path):
    zst_files = [f for f in os.listdir(folder_path) if f.endswith(".zst")]
    performances = []
    session_labels = []

    for file_name in zst_files:
        path = os.path.join(folder_path, file_name)
        try:
            data = npu.load_zst_file(path)

            if hasattr(data, "Correct") and hasattr(data, "Miss") and hasattr(data, "Wrong"):
                nb_correct = np.sum(data.Correct == 1)
                nb_miss = np.sum(data.Miss == 1)
                nb_wrong = np.sum(data.Wrong == 1)
                total = nb_correct + nb_miss + nb_wrong

                if total > 0:
                    performance = nb_correct / total
                else:
                    performance = np.nan

                # Raccourcir les noms de session (ex: ATX_230403)
                condition = file_name.split('_')[1]
                date = file_name.split('_')[-2]
                short_name = f"{condition}_{date}"

                session_labels.append(short_name)
                performances.append(performance)
        except Exception as e:
            print(f"Erreur pour {file_name} : {e}")

    return session_labels, performances


# === Dossiers ===
sham_folder = "C:\\Users\\eropartz\\Desktop\\ISC-MJ\\DATA\\Attention_OnlyBehavior\\Alix\\Sham"
stim_folder = "C:\\Users\\eropartz\\Desktop\\ISC-MJ\\DATA\\Attention_OnlyBehavior\\Alix\\ATX"

# === Charger les données ===
sham_sessions, sham_perf = get_performances_from_folder(sham_folder)
stim_sessions, stim_perf = get_performances_from_folder(stim_folder)

# === Créer le graphe ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharey=True)

# Groupe SHAM
ax1.bar(sham_sessions, sham_perf, color='skyblue')
ax1.set_title("Performance d'Alix (Sham)")
ax1.set_ylabel("Performance")
ax1.set_ylim(0, 1)
ax1.grid(axis='y')
ax1.tick_params(axis='x', rotation=45)

# Groupe STIM
ax2.bar(stim_sessions, stim_perf, color='salmon')
ax2.set_title("Performance d'Alix (ATX)")
ax2.set_ylabel("Performance")
ax2.set_xlabel("Sessions")
ax2.set_ylim(0, 1)
ax2.grid(axis='y')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

####

# TEST BIOSTAT

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# Nettoyage des NaN
sham_perf = np.array(sham_perf)
stim_perf = np.array(stim_perf)

sham_perf = sham_perf[~np.isnan(sham_perf)]
stim_perf = stim_perf[~np.isnan(stim_perf)]

# === Test t de Student ===
t_stat, p_value = ttest_ind(sham_perf, stim_perf)

# === Graphique : Boxplot comparatif ===
plt.figure(figsize=(6, 6))
plt.boxplot([sham_perf, stim_perf], labels=["SHAM", "ATX"], patch_artist=True,
            boxprops=dict(facecolor="lightblue"), medianprops=dict(color="black"))

# Ajout de la moyenne par point
plt.plot([1]*len(sham_perf), sham_perf, 'o', color='blue', alpha=0.5)
plt.plot([2]*len(stim_perf), stim_perf, 'o', color='red', alpha=0.5)



# Ajout de la p-value
plt.title("Comparaison des performances (SHAM vs ATX)")
plt.ylabel("Performance (Correct / Total)")
plt.text(1.5, 0.95, f"p = {p_value:.4f}", ha='center', fontsize=12,
         bbox=dict(facecolor='white', edgecolor='black'))

plt.ylim(0, 1.05)
plt.grid(axis='y')
plt.tight_layout()
plt.show()