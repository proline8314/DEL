# path = "/data02/gtguo/DEL/data/dataset/ca9_ligands/0/7pom_poc.pdbqt"
# path = "/data03/gtguo/del/docking/sEH/8qmz_poc.pdbqt"
path = "/data03/gtguo/del/docking/CA2/6oe1_poc.pdbqt"

# delete lines begin with "END", "ROOT", "BRANCH" and "TORSDOF"
with open(path, "r") as f:
    lines = f.readlines()
with open(path, "w") as f:
    for line in lines:
        if not line.startswith("END") and not line.startswith("ROOT") and not line.startswith("BRANCH") and not line.startswith("TORSDOF"):
            f.write(line)