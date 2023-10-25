# container_detection

최종 3등

key idea
- 이미지가 1024x1024인데 object 사이즈가 너무 작아서 이미지를 4~5배 정도 키운뒤에 1024 1024 사이즈만큼 crop 하여 학습진행
- inference 진행시에도 이미지를 확대하여 multi scale(3배 / 4배 / 5배)로 1024x1024 사이즈에 stride 512로 overlap을 주며 output을 낸후에 결과값을 iou 기준으로 합쳐서 사용함
- 오브젝트가 워낙 작았기때문에 이미지를 좀 더 키우고 crop size도 좀 더 키워서 해보는게 성능이 더 높지 않았을까 생각됨
- mmrotate 관련하여 사용할일 있으면 참조하면 좋을 것 같고, output이 x,y,w,h,a(각도) 형태의 데이터를 8 coordinate(4개의 좌표)로 변형하는 함수도 참조해서 다시 쓸 일 있을듯(rotate object detection 관련..)
