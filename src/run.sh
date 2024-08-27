modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-VPSc8YTJ3HIMeQLFfC9FgG
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-r8fdtpxCVEHSAhH1TiimE2
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-3UiBtxZBchpjFS98BmdKKr
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-Rrbcb0Wo29QyU6p7ytt9mQ
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-SOclLWJ8Clm4vEKhumAtVw
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b2 --version v1 --mode pretrain