import click
import torch
from pathlib import Path


@click.command()
@click.argument(
    "weights_fp", type=click.Path(exists=True, dir_okay=False, readable=True)
)
@click.argument(
    "output_dir",
    type=click.Path(dir_okay=True, writable=True),
    default=".",
)
def main(weights_fp, output_dir):
    state_dict = torch.load(weights_fp, map_location="cpu")

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    backbone_state_dict = {}
    for key, value in state_dict.items():
        print(f"Processing key: {key}")
        if "backbone." in key:
            new_key = key.split("backbone.")[-1]
            print(f" --> : {new_key}")
            new_key = "backbone." + new_key
            print(f" --> Adding key: {new_key}")
            backbone_state_dict[new_key] = value


    # Créer le répertoire de sortie s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Chemin de sortie
    orig_model_name = Path(weights_fp).stem
    output_fp = Path(output_dir) / f"{orig_model_name}_backbone.pth"

    # Sauvegarder le nouveau state_dict
    torch.save(backbone_state_dict, output_fp)


if __name__ == "__main__":
    main()
