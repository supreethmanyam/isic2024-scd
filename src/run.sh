modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b1 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-fwy2zOVSsxDsiVtWBsNI5R
modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b1 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-gGUO6PLSb4Nylls93UzniX
modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b1 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-4g81xd1qOzTJ9ASBSNBi3Q
modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b1 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-P0EVyP0VV40vAp7kIduV88
modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b1 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-32RGFWgpg5Lj0svKmIG90A
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b1 --version v1 --mode trainmulti
