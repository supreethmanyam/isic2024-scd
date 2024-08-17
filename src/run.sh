modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-i4LtvIZjYqurJV9i21nWbb
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-AdUvWD8fXuFh06eMg6yZ20
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-E2M9RX1nzhiy8ipUpzbLa3
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-EalJWQyKgvOOQ3akJx8TSf
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-8cSOFqx8fj0GvvyFjWZyTY
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b1 --version v1 --mode pretrain

modal run -d isic2024_scd_app.py::finetune --model-name efficientnet_b1 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-WzP330c7zvW93DK0dUQOrp
modal run -d isic2024_scd_app.py::finetune --model-name efficientnet_b1 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-wsSMc51AxbFjut44fzeBkM
modal run -d isic2024_scd_app.py::finetune --model-name efficientnet_b1 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-sZiKBjsMOv7El8s2Tm1YJB
modal run -d isic2024_scd_app.py::finetune --model-name efficientnet_b1 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-EOkF3huegs1UkKG9fhXVXH
modal run -d isic2024_scd_app.py::finetune --model-name efficientnet_b1 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-Wvo6fm4GbCv9Nwuftqx9Mw
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b1 --version v1 --mode finetune