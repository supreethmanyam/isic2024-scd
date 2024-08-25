modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::train --model-name efficientnet_b2 --version v4 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-tBvThCCiEupm15KreM8i2S