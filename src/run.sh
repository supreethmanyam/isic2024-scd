modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 1
#Link:
modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 2
#Link:
modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-17FUZ096UEvjL0DF6aM9Xu
modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 4
#Link:
modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 5
#Link:
modal run -d isic2024_scd_app.py::upload_weights --model-name tf_efficientnet_b1_ns --version v1 --mode trainbinary
