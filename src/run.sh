modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-fC04gNGIYRm8HdmcEi4TBC
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-Bp0XySDUZvpHxD7zsPD5zE
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-7KQZbX0qWn9gGvqYuszlVG
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-R0bjtrQdFJQlf1OFZLwP9e
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-yn2KuHDxsbPrRSSq0kIja2
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b2 --version v1 --mode pretrain
