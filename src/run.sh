modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v3 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-1iEVQx1aOoPxuiVQ9xuB8q
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v3 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-FBPsKqfSRvvr9EfmL1OjXG
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v3 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-cGDpu8OK1gAlcEpkbfSU5f
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v3 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-i3hc3pgkKLlvCwbbwgi68E
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v3 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-9JqgoL8Syj4z83etnN7MWx
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b0 --version v3 --mode trainbinary
