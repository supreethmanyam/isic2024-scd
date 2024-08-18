modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-ofhCBzGAqJ5dfhyqrhZaKs
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-MFOH192jSbWwUISPAVgnZI
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-B1gWM8C8QWwU3fjqlX8X2h
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-UCZFe19943eDlbimoY3qK9
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b2 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-BpE48RPEwWwH6QCaTr6wKm
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b2 --version v1 --mode pretrain

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b3 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-kR8db7O37bRU7Q1QtXdKJK
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b3 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-ZVU5pThOj0SdqwX6ifKDCL
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b3 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-IUFesHrcU9jKwzQJXCYaGK
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b3 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-BNOyNQVqgHZAVfZ9DPuDcZ
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b3 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-MIppc8foWZjgMuHbcapkPT
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b3 --version v1 --mode pretrain