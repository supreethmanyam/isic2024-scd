modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v3 --fold 1
#Link: https://modal.com/apps/supreeth-manyam/main/ap-okZUnXmtwCGe5gRggk40oF
modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v2 --fold 2
#Link:
modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v2 --fold 3
#Link:
modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v2 --fold 4
#Link:
modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v2 --fold 5
#Link:
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b0 --version v2