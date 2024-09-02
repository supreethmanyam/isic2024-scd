modal run -d isic2024_scd_app.py::download_data

modal run -d isic2024_scd_app.py::trainmulti --model-name tf_efficientnet_b1_ns --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-xFYIhA3sYozw9vTqvzWpLB
modal run -d isic2024_scd_app.py::trainmulti --model-name tf_efficientnet_b1_ns --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-FnVitBxnuadvQPxrWMehAn
modal run -d isic2024_scd_app.py::trainmulti --model-name tf_efficientnet_b1_ns --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-pmOfWpCeR1aLi7Lcyxu0pw
modal run -d isic2024_scd_app.py::trainmulti --model-name tf_efficientnet_b1_ns --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-naFAALG1E3UbzYzXVsIgPj
modal run -d isic2024_scd_app.py::trainmulti --model-name tf_efficientnet_b1_ns --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-kqaaoFxQYzxZ0fQvN8e7sK
modal run -d isic2024_scd_app.py::upload_weights --model-name tf_efficientnet_b1_ns --version v1 --mode trainmulti

modal run -d isic2024_scd_app.py::trainmulti --model-name mobilevitv2_200 --version v1 --fold 1
#Link: https://modal.com/apps/supreethmanyam/main/ap-XmDhdaTUIhd82pqag9uhD9
modal run -d isic2024_scd_app.py::trainmulti --model-name mobilevitv2_200 --version v1 --fold 2
#Link: https://modal.com/apps/supreethmanyam/main/ap-QbsZJcZoTMgzcfYYvqm9NK
modal run -d isic2024_scd_app.py::trainmulti --model-name mobilevitv2_200 --version v1 --fold 3
#Link: https://modal.com/apps/supreethmanyam/main/ap-eY7ItU1o3eaMrhDweZTBXa
modal run -d isic2024_scd_app.py::trainmulti --model-name mobilevitv2_200 --version v1 --fold 4
#Link: https://modal.com/apps/supreethmanyam/main/ap-VJSHLfTnJYghn7WQFg7QA2
modal run -d isic2024_scd_app.py::trainmulti --model-name mobilevitv2_200 --version v1 --fold 5
#Link: https://modal.com/apps/supreethmanyam/main/ap-yS8AhbcYF3qRHvlHyN3VIp
modal run -d isic2024_scd_app.py::upload_weights --model-name mobilevitv2_200 --version v1 --mode trainmulti
