modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-Axd4onarqEdK4a4WLYQAkK
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-4oOShfd5xrsyaR6fFGTmMa
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-8hS7HqKuXoWi7lNYkAn7oO
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-0zfE5CF2jTDCVlzMQz6BDa
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-XUQv1XsGUlYhoE9HwWMyU8
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b2 --version v1 --mode pretrain