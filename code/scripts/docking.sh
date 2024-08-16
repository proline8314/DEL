file_dir=/data02/gtguo/DEL/data/dataset/ca9_ligands_small
config_fpath=/data02/gtguo/DEL/data/dataset/ca9_ligands_small/config.txt
vina_fpath=/opt/vina/bin/vina
start_idx=4474
end_idx=4970

for idx in $(seq $start_idx $end_idx);
do
    cd $file_dir/$idx
    ligand_fpath=$file_dir/$idx/ligand.pdbqt
    out_fpath=$file_dir/$idx/ligand_out.pdbqt
    log_fpath=$file_dir/$idx/log.txt
    echo "Docking ligand $idx"
    $vina_fpath --config $config_fpath --ligand $ligand_fpath --out $out_fpath --log $log_fpath
done
