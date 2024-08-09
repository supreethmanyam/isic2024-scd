modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v5 --fold 1
#Link: https://modal.com/apps/supreeth-manyam/main/ap-yDwPhwCBVA1nCdW7XwPmkH
modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v5 --fold 2
#Link:
modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v5 --fold 3
#Link:
modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v5 --fold 4
#Link:
modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v5 --fold 5
#Link:
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b0 --version v5