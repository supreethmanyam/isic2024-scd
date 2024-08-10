modal run -d isic2024_scd_app.py::train --model-name mobilevitv2_200 --version v1 --fold 1
#Link: https://modal.com/apps/supreeth-manyam/main/ap-jwHthlKdmRTzGvzIcgQ1Eo
modal run -d isic2024_scd_app.py::train --model-name mobilevitv2_200 --version v3 --fold 2
#Link:
modal run -d isic2024_scd_app.py::train --model-name mobilevitv2_200 --version v3 --fold 3
#Link:
modal run -d isic2024_scd_app.py::train --model-name mobilevitv2_200 --version v3 --fold 4
#Link:
modal run -d isic2024_scd_app.py::train --model-name mobilevitv2_200 --version v3 --fold 5
#Link:
modal run -d isic2024_scd_app.py::upload_weights --model-name mobilevitv2_200 --version v3