modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 1
#Link: https://modal.com/apps/supreeth-manyam/main/ap-MfMy22RJT23JzJwC7knxge
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 2
#Link: https://modal.com/apps/supreeth-manyam/main/ap-FFDBbYHYlIfcMDh8MMyeaO
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 3
#Link: https://modal.com/apps/supreeth-manyam/main/ap-R5j4QRKzcPGQyXazGCjlMz
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 4
#Link: https://modal.com/apps/supreeth-manyam/main/ap-beDLUzERSePhE5neFBwKKZ
modal run -d isic2024_scd_app.py::pretrain --model-name efficientnet_b1 --version v1 --fold 5
#Link: https://modal.com/apps/supreeth-manyam/main/ap-SbMGHAjP9pBrkQua1Er5Jt
modal run -d isic2024_scd_app.py::upload_weights --model-name efficientnet_b1 --version v1 --mode pretrain

modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v2 --fold 1
#Link: https://modal.com/apps/supreeth-manyam/main/ap-G7CNuUMLJyxL33KDYDR3f4
modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v2 --fold 2
#Link: https://modal.com/apps/supreeth-manyam/main/ap-7Zyxq9H9YFAkijq39NITX2
modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v2 --fold 3
#Link: https://modal.com/apps/supreeth-manyam/main/ap-0NCsSyguChNN1kWJ5O0WRj
modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v2 --fold 4
#Link: https://modal.com/apps/supreeth-manyam/main/ap-uR576SBHszBHwSmZOaRmuL
modal run -d isic2024_scd_app.py::pretrain --model-name mobilevitv2_200 --version v2 --fold 5
#Link: https://modal.com/apps/supreeth-manyam/main/ap-gbMRSiyJyyb8bhLkmwcsuw
modal run -d isic2024_scd_app.py::upload_weights --model-name mobilevitv2_200 --version v2 --mode pretrain