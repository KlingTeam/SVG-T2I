## Training Scripts of SVG-T2I-Autoencoder

### Installation

[Taming-Transformers](https://github.com/CompVis/taming-transformers?tab=readme-ov-file) is needed for training. 
    
    Get it by running:
    ```
    git clone https://github.com/CompVis/taming-transformers.git
    cd taming-transformers
    pip install -e .
    ```

    Then modify ``./taming-transformers/taming/data/utils.py`` to meet torch 2.x:
    ```
    export FILE_PATH=./taming-transformers/taming/data/utils.py
    sed -i 's/from torch._six import string_classes/from six import string_types as string_classes/' "$FILE_PATH"
    ```


### Train

1. Modify training config as you need.

2. Run training by:

  ```bash
  bash run_train.sh <GPU NUM> configs/pure/svg_autoencoder_P_dd_M_IN_stage1_bs64_256_gpu1_forTest
  # example 
  bash run_train.sh 1 configs/pure/svg_autoencoder_P_dd_M_IN_stage1_bs64_256_gpu1_forTest
  ````

  * **Output:** Results will be saved in `autoencoder/logs`.
  * **Note:** You can modify training hyperparameters and output paths directly inside `run_train.sh` or the configuration YAML file.


### Acknowledgement

SVG-T2I's autoencoder training is mainly built upon [VA-VAE](https://github.com/hustvl/LightningDiT/) and [LDM](https://github.com/CompVis/latent-diffusion/tree/main). Thanks for the great work!
