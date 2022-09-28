# ICU Mortality Prediction
## 설명
MIMIC-III의 ICUSTAYS, CHARTEVENTS, ADMISSIONS 데이터를 바탕으로 Intensive Care Unit (ICU) 치료 중 사망 환자를 예측합니다.
ICU에 1 ~ 2일 사이의 기간동안(min_los: 1, max_los: 2) 입원한 환자의 처음 3시간 데이터만 가지고 사망을 예측합니다.
CHARTEVENTS에 기록된 ICU에 입원동안 발생한 이벤트(e.g. 약물 투여, heart rate 측정 등)를 바탕으로 feature를 구성하고 GRU 모델을 이용해 사망을 예측합니다.
이때 CHARTEVENTS의 이벤트 종류를 나타내는 ITEMID, 각 ITEMID에 해당하는 수치인 VALUENUM을 이용합니다.
추가로 예측 대상이 되는 환자별로 ADMISSIONS에 있는 ETHNICITY (인종), ADMISSION_TYPE (내원 타입), DIAGNOSIS (진단) 특징을 사용합니다.
본 실험에서 사용한 데이터는 민감한 의료 데이터이기 때문에 접근 권한이 없으면 데이터를 볼 수 없습니다.
따라서 실험에 사용한 데이터를 공개하지는 않겠습니다.
<br><br><br>

## 모델 종류
* ### GRU
    GRU를 이용하여 CHARTEVENT의 timestamp별로 발생한 이벤트를 featuring하여 환자 사망을 예측합니다.
<br><br><br>


## 사용 데이터
* [MIMIC-III](https://mimic.mit.edu/docs/iii/)
<br><br><br>


## 사용 방법
* ### 학습 방법
    학습을 시작하기 위한 argument는 4가지가 있습니다.<br>
    * [-d --device] {cpu, gpu}, **필수**: 학습을 cpu, gpu로 할건지 정하는 인자입니다.
    * [-m --mode] {train, test}, **필수**: 학습을 시작하려면 train, 최종 학습된 모델을 가지고 있어서 학습된 모델의 성능을 평가하고 싶으면 test 모드를 사용해야 합니다. test 모드를 사용할 경우, [-n, --name] 인자가 **필수**입니다.
    * [-c --cont] {1}, **선택**: 학습이 중간에 종료가 된 경우 다시 저장된 모델의 체크포인트 부분부터 학습을 시작할 수 있습니다. 이 인자를 사용할 경우 -m train 이어야 합니다. 
    * [-n --name] {name}, **선택**: 이 인자는 -c 1 혹은 -m test 경우 사용합니다.
    중간에 다시 불러서 학습을 할 경우 모델의 이름을 입력하고, test, inference를 할 경우에도 확인할 모델의 이름을 입력해주어야 합니다(최초 학습시 config.json에서 정한 모델의 이름의 폴더가 형성되고 그 폴더 내부에 모델 및 모델 파라미터가 json 파일로 형성 됩니다).<br><br>

    터미널 명령어 예시<br>
    * 최초 학습 시
        ```
        python3 main.py -d cpu -m train
        ```
    * 중간에 중단 된 모델 이어서 학습 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/config.json이 아닌, base_path/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.
        ```
        python3 main.py -d gpu -m train -c 1 -n {model_name}
        ```
    * 최종 학습 된 모델의 test set에 대한 성능 결과를 확인할 시
        <br>주의사항: config.json을 수정해야하는 일이 발생 한다면 base_path/config.json이 아닌, base_path/model/{model_name}/{model_name}.json 파일을 수정해야 수정사항이 반영됩니다.
        ```
        python3 main.py -d cpu -m test -n {model_name}
        ```
    <br><br>

* ### 모델 학습 조건 설정 (config.json)
    * **주의사항: 최초 학습 시 config.json이 사용되며, 이미 한 번 학습을 한 모델에 대하여 parameter를 바꾸고싶다면 base_path/model/{model_name}/{model_name}.json 파일을 수정해야 합니다.**
    * base_path: 학습 관련 파일이 저장될 위치
    * model_name: 학습 모델이 저장될 파일 이름 설정. 모델은 base_path/model/{model_name}/{model_name}.pt 로 저장.
    * loss_data_name: 학습 시 발생한 loss data를 저장하기 위한 이름 설정. base_path/loss/{loss_data_name}.pkl 파일로 저장. 내부에 중단된 학습을 다시 시작할 때, 학습 과정에 발생한 loss 데이터를 그릴 때 등 필요한 데이터를 dictionary 형태로 저장.
    * baseInfo_dim: ADMISSION에 있는 ETHNICITY, ADMISSION_TYPE, DIAGNOSIS 값을 임베딩할 차원.
    * model_hidden_dim: GRU 모델의 hidden dimension.
    * num_layers: GRU 모델의 레이어 수.
    * dropout: GRU 모델의 dropout 비율.
    * batch_size: batch size 지정.
    * epochs: 학습 epoch 설정.
    * lr: 학습 learning rate.
    * early_stop_criterion: Test set의 최대 accuracy 내어준 학습 epoch 대비, 설정된 숫자만큼 epoch이 지나도 나아지지 않을 경우 학습 조기 종료.    
    * min_los: ICU 환자 중 최소 LOS (length of stay) 기간 지정.
    * max_los: ICU 환자 중 최대 LOS (length of stay) 기간 지정.
    * max_chartevent_time: CHARTEVENTS 중 feature로 사용할 데이터의 최대 시간(e.g. 3이면 최초 입원 시간 이후 3시간의 데이터만 사용).
    * max_seq: CHARTEVENTS 중 사용할 최대 timestamp 개수.
    * itemid_topk: CHARTEVENTS에서 feature로 사용할 ITEMID top-k.
    * diag_topk: ADMISSIONS에서 feature로 사용할 DIAGNOSIS top-k.
    <br><br><br>


## 결과
* ### Test Set 결과
    * AUROC: 0.8216
    * AUPRC: 0.2718
        
<br><br><br>