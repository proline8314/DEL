file_dir=/data02/gtguo/DEL/data/dataset/ca9_ligands_small
pocket_fpath=/data02/gtguo/DEL/data/dataset/ca9_ligands_small/7pom_poc.pdbqt
num_splits=4971

# Path: DEL/code/scripts/docking_protocal.sh
# use openbabel to convert ligands to pdbqt

for i in $(seq 0 $num_splits); do
    echo "Processing split $i"
    cd $file_dir/$i/
    obabel ligand.pdb -O ligand.pdbqt
    cp $pocket_fpath .
done
