modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name seresnet50 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-HPCUfp93i17440yjukB1YO
modal run -d isic2024_scd_app.py::pretrain --model-name seresnet50 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-LPUXKsVovw0AiRMpc5lhbA
modal run -d isic2024_scd_app.py::pretrain --model-name seresnet50 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-ELWdPLOpNc7nRWILo9IIVb
modal run -d isic2024_scd_app.py::pretrain --model-name seresnet50 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-Pdzabs2BsIFJVQEh8F5gbF
modal run -d isic2024_scd_app.py::pretrain --model-name seresnet50 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-CAdJUFpjGgESEkLN8wVvXA
modal run -d isic2024_scd_app.py::upload_weights --model-name seresnet50 --version v1 --mode pretrain


#modal run -d isic2024_scd_app.py::pretrain --model-name tf_efficientnetv2_s --version v1 --fold 1
##Link: https://modal.com/apps/supreethmanyam/main/ap-N0hh3dj2LRb5qqtRtV8m3J
#modal run -d isic2024_scd_app.py::pretrain --model-name tf_efficientnet_b4 --version v1 --fold 1
##Link: https://modal.com/apps/supreethmanyam/main/ap-Lk82o7VMD2sR245JuxKOjP