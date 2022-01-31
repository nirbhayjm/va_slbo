import wandb


def init_wandb(wandb_name, project_name, config={}, group=None, resume=False):
    config = dict(config).copy()

    wandb_run = wandb.init(
        project=project_name, config=config, group=group, resume=resume
    )

    wandb_run_name = wandb_name
    wandb_run.name = wandb_run_name
    wandb_run.save()
