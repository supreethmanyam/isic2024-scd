# isic2024-scd

### Setup

1. Create a conda environment. Modify the conda environment as needed.
   ```console
   conda create -n isic2024_scd python=3.10
   ```
2. Activate the conda environment
   ```console
   conda activate isic2024_scd
   ```
3. Install the required packages
   ```console
   poetry install --no-root --with dev,lint
   ```