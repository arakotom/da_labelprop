#!/bin/bash
#SBATCH --job-name=ssl_iid          # nom du job
# Il est possible d'utiliser une autre partition que celle par défaut
# en activant l'une des 5 directives suivantes :
##SBATCH -C v100-16g                 # decommenter pour reserver uniquement des GPU V100 16 Go
##SBATCH -C v100-32g                 # decommenter pour reserver uniquement des GPU V100 32 Go
##SBATCH --partition=gpu_p2          # decommenter pour la partition gpu_p2 (GPU V100 32 Go)
##SBATCH -C a100                     # decommenter pour la partition gpu_p5 (GPU A100 80 Go)
##SBATCH -C h100                     # decommenter pour la partition gpu_p6 (GPU H100 80 Go)
# Ici, reservation de 10 CPU (pour 1 tache) et d'un GPU sur un seul noeud :
#SBATCH --nodes=1                    # on demande un noeud
#SBATCH --ntasks-per-node=1          # avec une tache par noeud (= nombre de GPU ici)
#SBATCH --gres=gpu:1                 # nombre de GPU par noeud (max 8 avec gpu_p2, gpu_p5)
# Le nombre de CPU par tache doit etre adapte en fonction de la partition utilisee. Sachant
# qu'ici on ne reserve qu'un seul GPU (soit 1/4 ou 1/8 des GPU du noeud suivant la partition),
# l'ideal est de reserver 1/4 ou 1/8 des CPU du noeud pour la seule tache:
#SBATCH --cpus-per-task=10           # nombre de CPU par tache (1/4 des CPU du noeud 4-GPU V100)
##SBATCH --cpus-per-task=3           # nombre de CPU par tache pour gpu_p2 (1/8 des CPU du noeud 8-GPU V100)
##SBATCH --cpus-per-task=8           # nombre de CPU par tache pour gpu_p5 (1/8 des CPU du noeud 8-GPU A100)
##SBATCH --cpus-per-task=24           # nombre de CPU par tache pour gpu_p6 (1/4 des CPU du noeud 4-GPU H100)
# /!\ Attention, "multithread" fait reference à l'hyperthreading dans la terminologie Slurm
#SBATCH --hint=nomultithread         # hyperthreading desactive

#SBATCH --time=20:00:00              # temps maximum d'execution demande (HH:MM:SS)
#SBATCH --qos=qos_gpu-t3

##SBATCH --time=48:00:00              # temps maximum d'execution demande (HH:MM:SS)
##SBATCH --qos=qos_gpu-t4 


#SBATCH --output=out_%j.log      # nom du fichier de sortie
#SBATCH --error=out_%j.log       # nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH -A jrw@v100
# Nettoyage des modules charges en interactif et herites par defaut

module purge
 
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p5"
# pour avoir acces aux modules compatibles avec cette partition
#module load arch/a100
# Decommenter la commande module suivante si vous utilisez la partition "gpu_p6"
# pour avoir acces aux modules compatibles avec cette partition
#module load arch/h100
 
 
# Chargement des modules
module load pytorch-gpu/py3/2.0.0

 
# Echo des commandes lancees
set -x
 
# Pour les partitions "gpu_p5" et "gpu_p6", le code doit etre compile avec les modules compatibles
# avec la partition choisie

#{data:} {algo:} {bag_size} {nb_class_in_bag} {seed:}

if [ "$1" -eq 0 ]; then

    python -u expe.py  --data $2  --algo $3 --seed $7 --dep_sample 1 \
                    --bag_size $4 --source_target $5 --i_param $6> ./out/out_dep_$2_$3_$5_$4_$7_$1.log
fi 

if [ "$1" -eq 1 ]; then

    python -u expe.py  --data $2  --algo $3 --seed $7 --dep_sample 0 \
                    --bag_size $4 --source_target $5 --i_param $6> ./out/out_iid_$2_$3_$5_$4_$7_$1.log
fi 

