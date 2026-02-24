## Training Instructions
Example:

`config`: specifies config yaml file name in ./config folder

`seed`: specifies seed to run training run with

`neptune` activates neptune logging

    python -m scripts.train_agent --config train --seed 42 --neptune

## Inference Instructions
Example:

`run`: expects a folder name within ./runs folder, which in turn contains *actor* and *critic* weights in addition to model configurations

`latest`: runs the latest locally trained model in ./runs folder

    python -m scripts.inference --run train_seed42_2026-02-24_09-19-44

OR

    python -m scripts.inference --run latest