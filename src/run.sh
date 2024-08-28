modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v3 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-Exs4vues2hXszLlOVinWI5
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v3 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-zZOb295Aw6psaQFCLDNFbU
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v3 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-uXOt4WQrTHkhpt2p6gWZcD
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v3 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-KP8ULbM3wU5KnrpHv4HX3S
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v3 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-WFwr1MNl7LeyenbfSno0Ai
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b2 --version v3 --mode pretrain