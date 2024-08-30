modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v2 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-2Mh0r73JNaAQfxL7TCq5QH
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v2 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-3zwTFfx1jLTC3gBFwTPtW9
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v2 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-Suehj25EuRzmDWortpWghd
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v2 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-b1pHyIoCrmYnvdleKv0Vqv
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v2 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-UqAPTUMPPhjK7q472HpBdQ
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b0 --version v2 --mode trainbinary
