modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainbinary --model-name mobilevitv2_200 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-523S9zvmuIwBNhNmtVNOqw
modal run -d isic2024_scd_app.py::trainbinary --model-name mobilevitv2_200 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-H4JozfrGM4TVDKGG1a3s2J
modal run -d isic2024_scd_app.py::trainbinary --model-name mobilevitv2_200 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-KxATJgVWK662UUxZkJX08i
modal run -d isic2024_scd_app.py::trainbinary --model-name mobilevitv2_200 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-CmqIzgSbtpAgkVnlCVpwIT
modal run -d isic2024_scd_app.py::trainbinary --model-name mobilevitv2_200 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-h8A11Hd03xse38MktBdQXj
modal run -d isic2024_scd_app.py::upload_weights --model-name mobilevitv2_200 --version v1 --mode trainbinary

modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-G96h7IBg8hZycGbJsI0ijx
modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-9js9z85pB4oNhFUVVLDaxb
modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-sVHpARkp82UbNWVFarnVwo
modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-iuqodFu6AHEdpf899b5Xxk
modal run -d isic2024_scd_app.py::trainbinary --model-name tf_efficientnet_b1_ns --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-ZQdVfYlCktmWRNzfapvNAx
modal run -d isic2024_scd_app.py::upload_weights --model-name tf_efficientnet_b1_ns --version v1 --mode trainbinary
