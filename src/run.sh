modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-aPtEE4p16qra1LOgWRAwJR
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-eqVkCrrGQFhnVbh0Q9aXQP
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-u3LNsR9m7qSaiFgwvtP59y
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-jZ1OlwDN0mNDJn7SPs0wM7
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b0 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-sFQL9bDzZJFfoiccLDmgDd
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b0 --version v1 --mode trainbinary
