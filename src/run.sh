modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v6 --fold 1 --ext "2020,2019" --out-dim 9 --use-meta
https://modal.com/supreeth-manyam/main/apps/ap-le2wHSzGFcSYdLCwurh9yp
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v6 --fold 2 --ext "2020,2019" --out-dim 9 --use-meta
https://modal.com/supreeth-manyam/main/apps/ap-zopE6iWUKfHRPY7Zjpj7Yx
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v6 --fold 3 --ext "2020,2019" --out-dim 9 --use-meta
https://modal.com/supreeth-manyam/main/apps/ap-7Ruc5r8K4ICsIrD4ZOi3cO
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v6 --fold 4 --ext "2020,2019" --out-dim 9 --use-meta
https://modal.com/supreeth-manyam/main/apps/ap-LeI9KuExPbdWvbA6WKwNkS
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v6 --fold 5 --ext "2020,2019" --out-dim 9 --use-meta
https://modal.com/supreeth-manyam/main/apps/ap-YtNxzkp9yFrtYVJE25HQ3W
modal run -d isic2024_scd_app.py::upload_weights --model-name resnet18 --version v6 --out-dim 9 --use-meta

modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v7 --fold 1 --ext "2020,2019" --out-dim 9
https://modal.com/supreeth-manyam/main/apps/ap-FnhIkaa5oedtcOZec2VFMh
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v7 --fold 2 --ext "2020,2019" --out-dim 9
https://modal.com/supreeth-manyam/main/apps/ap-iHSYqGLsowpaVlg5CJavnj
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v7 --fold 3 --ext "2020,2019" --out-dim 9
https://modal.com/supreeth-manyam/main/apps/ap-KemzFucXDxD4dAcAEIv5lT
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v7 --fold 4 --ext "2020,2019" --out-dim 9
https://modal.com/supreeth-manyam/main/apps/ap-jQLrl1dyhrKtedJEMFUwAU
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v7 --fold 5 --ext "2020,2019" --out-dim 9
https://modal.com/supreeth-manyam/main/apps/ap-ZI1nltVMhXLeAEWq6kHooT
modal run -d isic2024_scd_app.py::upload_weights --model-name resnet18 --version v7 --out-dim 9


#modal run -d isic2024_scd_app.py::train --model-name efficientnet_b3 --version v1 --fold 1 --ext "2020,2019" --out-dim 9 --use-meta
#https://modal.com/supreeth-manyam/main/apps/ap-Q854jCNVgWIARaXnUv9zYE
#modal run -d isic2024_scd_app.py::train --model-name efficientnet_b3 --version v1 --fold 2 --ext "2020,2019" --out-dim 9 --use-meta
#https://modal.com/supreeth-manyam/main/apps/ap-wGUsKspJRcDPXyZ9YBQsNL
#modal run -d isic2024_scd_app.py::train --model-name efficientnet_b3 --version v1 --fold 3 --ext "2020,2019" --out-dim 9 --use-meta
#https://modal.com/supreeth-manyam/main/apps/ap-j4vPq2sQQIu52Q4rFS3ijN
#modal run -d isic2024_scd_app.py::train --model-name efficientnet_b3 --version v1 --fold 4 --ext "2020,2019" --out-dim 9 --use-meta
#https://modal.com/supreeth-manyam/main/apps/ap-QLCD5IpQ1XG6Xws45nILxV
#modal run -d isic2024_scd_app.py::train --model-name efficientnet_b3 --version v1 --fold 5 --ext "2020,2019" --out-dim 9 --use-meta
#https://modal.com/supreeth-manyam/main/apps/ap-Kycti8h6Juj1iAZZ2lIiEN
#modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b3 --version v1 --out-dim 9 --use-meta

modal run -d isic2024_scd_app.py::evaluate --model-name efficientnet_b3 --version v1 --fold 1 --epoch 5 --out-dim 9 --use-meta
modal run -d isic2024_scd_app.py::evaluate --model-name efficientnet_b3 --version v1 --fold 2 --epoch 1 --out-dim 9 --use-meta
modal run -d isic2024_scd_app.py::evaluate --model-name efficientnet_b3 --version v1 --fold 3 --epoch 2 --out-dim 9 --use-meta
modal run -d isic2024_scd_app.py::evaluate --model-name efficientnet_b3 --version v1 --fold 4 --epoch 5 --out-dim 9 --use-meta
modal run -d isic2024_scd_app.py::evaluate --model-name efficientnet_b3 --version v1 --fold 5 --epoch 3 --out-dim 9 --use-meta
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b3 --version v1 --out-dim 9


# Resnet18 6 epochs without meta with external data 9 classes
# Fold1
https://modal.com/supreeth-manyam/main/apps/ap-s2UhnmT0Ww9aficOfXdppL
# Fold2
https://modal.com/supreeth-manyam/main/apps/ap-2G7QbUkO0pgKz8FD9HqufR
# Fold3
https://modal.com/supreeth-manyam/main/apps/ap-RXFYcO4aVeOBGvLUNDRLCi
# Fold4
https://modal.com/supreeth-manyam/main/apps/ap-EAW0os0NWNh9awXZKc7q5f
# Fold5
https://modal.com/supreeth-manyam/main/apps/ap-0yb8sYxrfhjGNt4ZZMkDBj