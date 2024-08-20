modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name medmamba --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-RT9OJAnhIbPGn76kJHwrA5
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-O3lsII9jr9KRtfC1RD8XT6
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-GzJcU9GdfVEx0s7EwkCJSc
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-o8elgg5fLjhBZ1qtcaJltv
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-j4aI5r7ThqhpuFTWTolLhI
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b2 --version v1 --mode pretrain