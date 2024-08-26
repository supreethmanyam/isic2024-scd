modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 1
#Link: https://modal.com/logs/supreethmanyam/main/ap-M7jzdRZQwjMMTCXtUx46bN
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 2
#Link: https://modal.com/logs/supreethmanyam/main/ap-LHFGBYKygfzFy23uyl6lDK
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 3
#Link: https://modal.com/logs/supreethmanyam/main/ap-q6IWtLuO0o1S6mHAHwIlNp
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 4
#Link: https://modal.com/logs/supreethmanyam/main/ap-n340evaPIfU4Y26ThAArvR
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 5
#Link: https://modal.com/logs/supreethmanyam/main/ap-dRB1BUeVNBQpXN9lMd5KTo
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b2 --version v2 --mode pretrain