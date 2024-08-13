modal run -d isic2024_scd_app.py::train --model-name efficientnet_b0 --version v7 --fold 1
#Link:
modal run -d isic2024_scd_app.py::train --model-name resnet50 --version v1 --fold 2
#Link:
modal run -d isic2024_scd_app.py::train --model-name resnet50 --version v1 --fold 3
#Link:
modal run -d isic2024_scd_app.py::train --model-name resnet50 --version v1 --fold 4
#Link:
modal run -d isic2024_scd_app.py::train --model-name resnet50 --version v1 --fold 5
#Link:
modal run -d isic2024_scd_app.py::upload_weights --model-name resnet50 --version v1