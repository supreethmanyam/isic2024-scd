modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v4 --fold 1 --ext "2020,2019" --out-dim 2
https://modal.com/supreeth-manyam/main/apps/ap-lFSWj3NSwxvFFRM1D3F6z1
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v4 --fold 2 --ext "2020,2019" --out-dim 2
https://modal.com/supreeth-manyam/main/apps/ap-w3DhNCKXOLQCZHd91Y3FVR
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v4 --fold 3 --ext "2020,2019" --out-dim 2
https://modal.com/supreeth-manyam/main/apps/ap-aWmDl0QtyV6f9ArYzg8L39
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v4 --fold 4 --ext "2020,2019" --out-dim 2
https://modal.com/supreeth-manyam/main/apps/ap-aEn8uh2zOtqZ9XW790AqO5
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v4 --fold 5 --ext "2020,2019" --out-dim 2
https://modal.com/supreeth-manyam/main/apps/ap-rc840aV3gljiTQMgug0qQS
modal run -d isic2024_scd_app.py::upload_weights --model-name resnet18 --version v4

modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v5 --fold 1 --ext "2020,2019" --out-dim 9
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v5 --fold 2 --ext "2020,2019" --out-dim 9
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v5 --fold 3 --ext "2020,2019" --out-dim 9
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v5 --fold 4 --ext "2020,2019" --out-dim 9
modal run -d isic2024_scd_app.py::train --model-name resnet18 --version v5 --fold 5 --ext "2020,2019" --out-dim 9
modal run -d isic2024_scd_app.py::upload_weights --model-name resnet18 --version v5