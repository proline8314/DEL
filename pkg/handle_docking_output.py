import os

import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


def handle_docking_log_file(file):
    """
    Extract best vina score from the docking log file
    log example:
    #################################################################
    # If you used AutoDock Vina in your work, please cite:          #
    #                                                               #
    # O. Trott, A. J. Olson,                                        #
    # AutoDock Vina: improving the speed and accuracy of docking    #
    # with a new scoring function, efficient optimization and       #
    # multithreading, Journal of Computational Chemistry 31 (2010)  #
    # 455-461                                                       #
    #                                                               #
    # DOI 10.1002/jcc.21334                                         #
    #                                                               #
    # Please see http://vina.scripps.edu for more information.      #
    #################################################################

    Detected 32 CPUs
    WARNING: at low exhaustiveness, it may be impossible to utilize all CPUs
    Reading input ... done.
    Setting up the scoring function ... done.
    Analyzing the binding site ... done.
    Using random seed: 2055148987
    Performing search ... done.
    Refining results ... done.

    mode |   affinity | dist from best mode
        | (kcal/mol) | rmsd l.b.| rmsd u.b.
    -----+------------+----------+----------
    1         -6.9      0.000      0.000
    2         -6.5      2.750      4.293
    3         -6.4      5.248      7.018
    4         -6.4      4.309      9.166
    Writing output ... done.
    """
    with open(file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if "mode |   affinity | dist from best mode" in line:
                try:
                    vina_score = float(lines[i+3].split()[1])
                    return -vina_score
                except:
                    return None
    return None

dir_path = "/data02/gtguo/DEL/data/dataset/ca9_ligands_small"
is_active_fname = "is_active.npy"

is_active = np.load(os.path.join(dir_path, is_active_fname))

idx_list = []
vina_score_list = []
for i, active in tqdm(enumerate(is_active)):
    file = os.path.join(dir_path, f"{i}/log.txt")
    if os.path.exists(file):
        vina_score = handle_docking_log_file(file)
        if vina_score is not None:
            idx_list.append(i)
            vina_score_list.append(vina_score)

# calculate enrichment factor
idx_list = np.array(idx_list)
vina_score_list = np.array(vina_score_list)
is_active = is_active[idx_list]

def enrichment_factor(y_true, y_pred, cutoff=0.01):
    # vina score the lower the better
    y_true = y_true[y_pred.argsort()[::-1]]
    return y_true[:int(len(y_true)*cutoff)].mean() / y_true.mean()

print(enrichment_factor(is_active, vina_score_list, 0.001))
print(enrichment_factor(is_active, vina_score_list, 0.01))
print(enrichment_factor(is_active, vina_score_list, 0.05))
print(enrichment_factor(is_active, vina_score_list, 0.1))
print(roc_auc_score(is_active, vina_score_list))
