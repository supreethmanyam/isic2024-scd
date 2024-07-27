modal run -d isic2024_scd_app.py --model-name resnet18 --version v1 --fold 1
modal run -d isic2024_scd_app.py --model-name resnet18 --version v1 --fold 2
modal run -d isic2024_scd_app.py --model-name resnet18 --version v1 --fold 3
modal run -d isic2024_scd_app.py --model-name resnet18 --version v1 --fold 4
modal run -d isic2024_scd_app.py --model-name resnet18 --version v1 --fold 5
modal run -d isic2024_scd_app.py::upload_weights --model-name resnet18 --version v1
