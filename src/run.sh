modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-u2Y7NirE7RqDxkslqqc0x2
modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-bkQbwIz96iD7iAuWTAECuu
modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-d5CD0AJ9m2creQzDvJKBgd
modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-tUI4Xzk22oqg45NKpz241b
modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-YBcNFi0cX9kTYOHczrA4Y5
modal run -d isic2024_scd_app.py::upload_weights --model-name mobilevitv2_200 --version v1 --mode pretrain