# Done

# TODO
- [] JTT : trouver les bons HP pour chaque dataset (en utilisant HBO)
- Ecrire la partie experimentale du papier
- [] Experiment plan to send to Lisheng
- [] bien distinguer et logger la mean_acc ET la mean_group_acc (faire la distiction et se mettre au clair sur l'objectif d'optimisation
- [] dataset vraiemnt tres petits avec k=2 et mu=0.05 (10 exemples pour les groupes de minority  classes)
    1000/2(train)/20(minority)/3(y_sampling)
- Question : 
- [] Qu'est ce que la regularisation sur validation set evoquée dans le papier ?
- [] mu varying ?


# Study
+ maybe do the 2,4,8,16
+ conclusion du papier ?
+ influence of K as mu
+ 

Remarque :
- [] Les methodes suby et rwy qui sous sample les classent devraient pas performées vraiment differement sur un dataset equilibré => biaser notre dataset en terme de classe ?
- [] JTT : tres sensible aux hyper parametres!
- [] probleme de maximiser la relative accuracy : la mean accuracy peut etre presque à 0 et la relative accuracy a 1 (avantage de la worst acc c'est que c'est un minorant la mean acc) 
- [] Quelle accuracy utiliser ? La mean_acc est en fait la group_mean_acc qui n'est pas pondérée : chaque groupe à la meme importance

#
### Installing dependencies

Easiest way to have a working environment for this repo is to create a conda environement with the following commands

```bash
conda env create -f environment.yaml
conda activate balancinggroups
```	

If conda is not available, please install the dependencies listed in the requirements.txt file.

### Download, extract and Generate metadata for datasets

This script downloads, extracts and formats the datasets metadata so that it works with the rest of the code out of the box.

```bash
python setup_datasets.py --download --data_path data
```

### Launch jobs

To reproduce the experiments in the paper on a SLURM cluster :

```bash
# Launching 1400 combo seeds = 50 hparams for 4 datasets for 7 algorithms
# Each combo seed is ran 5 times to compute error bars, totalling 7000 jobs
python train.py --data_path data --output_dir main_sweep --num_hparams_seeds 1400 --num_init_seeds 5 --partition <slurm_partition>
```

If you want to run the jobs localy, omit the --partition argument.

### Parse results

The parse.py script can generate all of the plots and tables from the paper. 
By default, it generates the best test worst-group-accuracy table for each dataset/method.
This script can be called while the experiments are still running. 

```bash
python parse.py main_sweep
```

## License

This source code is released under the CC-BY-NC license, included [here](LICENSE).
