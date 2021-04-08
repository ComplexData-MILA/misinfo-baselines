import wandb
from simple_slurm import Slurm
from omegaconf import OmegaConf


slurm = Slurm(
    array=range(0,22),
    cpus_per_task=4,
    job_name='sweep',
    gres='gpu:rtx8000:1',
    mem='24gb',
    output=f'logs/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
)

sweep_conf = OmegaConf.to_container(OmegaConf.load("af_fakenewsnet_sweep.yaml"))

ent = sweep_conf['entity']
proj = sweep_conf['project']

sweep_id = wandb.sweep(sweep_conf, project=proj, entity=ent)
print(sweep_id)
slurm.sbatch(f'wandb agent --count 1 {ent}/{proj}/{sweep_id}')