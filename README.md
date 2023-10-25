# container_detection

최종 3등

key idea
- 이미지가 1024x1024인데 object 사이즈가 너무 작아서 이미지를 4~5배 정도 키운뒤에 1024 1024 사이즈만큼 crop 하여 학습진행
- inference 진행시에도 이미지를 확대하여 multi scale로 output을 낸후에 결과값을 iou 기준으로 합쳐서 사용함
