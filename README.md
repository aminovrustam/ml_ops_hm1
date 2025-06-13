## Домашнее задание по ML Ops1
Загрузите проект в отдельную папку. В этой же папке создайте папки с именами "input" и "output", "train_data".

Загрузите файл "train.csv" в папку "train_data" из соревнования https://www.kaggle.com/competitions/teta-ml-1-2025/data

Для запуска докера введите код:

*docker build -t fraud_detector .*

и 

*docker run -it --rm -v ./input:/app/input \
                    -v ./output:/app/output \
                    fraud_detector*


Скопируйте файл "test.csv" из https://www.kaggle.com/competitions/teta-ml-1-2025/data в папку "input"

Готово. В папке "output" появятся файл сабмита, json-файл с топ 5 фичами и график распределения предсказаний модели.
