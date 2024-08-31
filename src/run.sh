modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v2 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-o68SnUk9sjczd7UvRpqk8c
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v2 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-n5pGHeImZSyQCdP9trGrjf
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v2 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-eumz9cl8GQJdWoKpISPF12
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v2 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-IvJS6HHgBHupN6TzdCUY3n
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v2 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-YGMXwg7hX4H5vEt4qTZC45
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b1 --version v2 --mode trainbinary
