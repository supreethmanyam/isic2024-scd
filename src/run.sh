modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name resnet18 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-rbafQ2YaFwmolztToLXGD7
modal run -d isic2024_scd_app.py::pretrain --model-name resnet18 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-xUpQ47gIv0LO0StBG7rQri
modal run -d isic2024_scd_app.py::pretrain --model-name resnet18 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-RjNOWdF64m8gTqVH1t4NL3
modal run -d isic2024_scd_app.py::pretrain --model-name resnet18 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-hwt1Su2sqnAoWaFqbjOOY9
modal run -d isic2024_scd_app.py::pretrain --model-name resnet18 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-xjya6ICZfh691oj77hXoU8
modal run -d isic2024_scd_app.py::upload_weights --model-name resnet18 --version v1 --mode pretrain