modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 1
#Link: https://modal.com/logs/supreethmanyam/main/ap-VE5l4eErZA6QKj57BKLfv9
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 2
#Link: https://modal.com/logs/supreethmanyam/main/ap-DOyPlj0RuMEKjL6IuHrsZv
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 3
#Link: https://modal.com/logs/supreethmanyam/main/ap-9k4LQrj4SVY5nhpMdEXuq1
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 4
#Link: https://modal.com/logs/supreethmanyam/main/ap-QTREL2sz1kiBM7cN4Yz81B
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v2 --fold 5
#Link: https://modal.com/logs/supreethmanyam/main/ap-3nSWTna7hGc6gL94n85Qm5
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b2 --version v2 --mode pretrain