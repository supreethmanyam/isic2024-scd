modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b0 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-Qjxj53LHP6W0D2M4awW6UZ
modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b0 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-o2swjZF5vD0wup06zbnq3P
modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b0 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-L2xRZPEka8oeWNzdEdZEi1
modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b0 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-DpeCjWR8j57Tjr2UKOEZRQ
modal run -d isic2024_scd_app.py::trainmulti --model-name efficientnet_b0 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-BNPUJ5RLk1XIGP1VYbAgRs
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b0 --version v1 --mode trainmulti


modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-WL2V1ShnH81QLQ1oEvLLr1
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-KTRP9BiVnioNgJud0XkHMW
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-Mbdh25y494F4FiMH5Lt4GH
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-aHF9YtY8KrgJiHFCij53gg
modal run -d isic2024_scd_app.py::trainbinary --model-name efficientnet_b1 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-ZPTQE8J4mYlcCxGiPqsgY5
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b1 --version v1 --mode trainbinary