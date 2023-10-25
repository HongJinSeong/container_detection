import cv2
import json
import numpy as np
import glob as _glob
import os
import random
import csv


csv.register_dialect(
    'mydialect',
    delimiter = ',',
    quotechar = '"',
    doublequote = True,
    skipinitialspace = True,
    lineterminator = '\r\n',
    quoting = csv.QUOTE_MINIMAL)


def csv2list(filename):
  lists=[]
  file=open(filename,"r")
  while True:
    line=file.readline().replace('\n','')
    if line:
      line=line.split(",")
      lists.append(line)
    else:
      break
  return lists

def glob(dir, pats, recursive=False):  # faster than match, python3 only
    pats = pats if isinstance(pats, (list, tuple)) else [pats]
    matches = []
    for pat in pats:
        matches += _glob.glob(os.path.join(dir, pat), recursive=recursive)
    return matches

## 라벨 save 코드 나중에 아래의 라벨 검증코드에 따라서 약간 수정을 거쳐야 할 것 같음
# all_lbls = glob('container_dataset/train_labels','*')
# random.shuffle(all_lbls)
#
# train_lbls = all_lbls[:int(len(all_lbls)*0.8)]
#
# valid_lbls = all_lbls[int(len(all_lbls)*0.8):]
#
# for lbl in train_lbls:
#     with open(lbl, 'r') as file:
#         data = json.load(file)
#     for data_row in data['features']:
#         INdata = list(map(str,map(int,(map(float, data_row['properties']['object_imcoords'].split(','))))))
#         INdata = ' '.join(INdata)
#         INdata +=' container 0'
#
#         with open("container_dataset/train_lbl/" + lbl.replace('\\', '/').split('/')[-1].replace('.json', '.txt'), "a") as f:
#             f.write(INdata+'\n')
#
# for lbl in valid_lbls:
#     with open(lbl, 'r') as file:
#         data = json.load(file)
#     for data_row in data['features']:
#         INdata = list(map(str,map(int,(map(float, data_row['properties']['object_imcoords'].split(','))))))
#         INdata = ' '.join(INdata)
#         INdata +=' container 0'
#
#         with open("container_dataset/valid_lbl/" + lbl.replace('\\', '/').split('/')[-1].replace('.json', '.txt'), "a") as f:
#             f.write(INdata+'\n')



all_imgs = glob('container_dataset/train_images','*.png')
all_lbls = glob('container_dataset/train_labels','*')

for i in range(len(all_imgs)):
    img = cv2.imread(all_imgs[i])
    with open(all_lbls[i], 'r') as file:
        data = json.load(file)
    for data_row in data['features']:
        pos_8 = list(map(int,(map(float, data_row['properties']['object_imcoords'].split(',')))))
        pos_8 = np.array([[pos_8[0:2]], [pos_8[2:4]], [pos_8[4:6]], [pos_8[6:8]]], np.int32)
        img = cv2.polylines(img, [pos_8], True, (255, 255, 255), 2)
        for j in range(pos_8.shape[0]):
            if j == 0:
                img = cv2.line(img, pos_8[j][0], pos_8[j][0], (255, 0, 0), thickness=2)
            if j == 1:
                img = cv2.line(img, pos_8[j][0], pos_8[j][0], (0, 255, 0), thickness=2)
            if j == 2:
                img = cv2.line(img, pos_8[j][0], pos_8[j][0], (0, 0, 255), thickness=2)
            if j == 3:
                img = cv2.line(img, pos_8[j][0], pos_8[j][0], (0, 255, 255), thickness=2)
    cv2.imwrite('checklabel_real_OK/' + all_imgs[i].replace('\\', '/').split('/')[-1],img)


############## mmrotate tutorial data검증
# all_imgs = glob('ssdd_tiny/images','*.png')
# train_lbls = glob('ssdd_tiny/train','*')
# val_lbls = glob('ssdd_tiny/val','*')
# all_lbls = train_lbls + val_lbls
# for i in range(len(all_lbls)):
#     img = cv2.imread('ssdd_tiny/images/' + all_lbls[i].replace('\\','/').split('/')[-1][:-4] + '.png')
#     lbl = csv2list(all_lbls[i])
#     for data_row in lbl:
#         pos_8 = list(map(int,data_row[0].split()[:8]))
#         pos_8 = np.array([[pos_8[0:2]], [pos_8[2:4]], [pos_8[4:6]], [pos_8[6:8]]], np.int32)
#         img = cv2.polylines(img, [pos_8], True, (255, 255, 255), 2)
#         for j in range(pos_8.shape[0]):
#             if j == 0:
#                 img = cv2.line(img, pos_8[j][0], pos_8[j][0], (255, 0, 0), thickness=2)
#             if j == 1:
#                 img = cv2.line(img, pos_8[j][0], pos_8[j][0], (0, 255, 0), thickness=2)
#             if j == 2:
#                 img = cv2.line(img, pos_8[j][0], pos_8[j][0], (0, 0, 255), thickness=2)
#             if j == 3:
#                 img = cv2.line(img, pos_8[j][0], pos_8[j][0], (0, 255, 255), thickness=2)

#     cv2.imwrite('check_mmrotate_tuto/' + all_lbls[i].replace('\\','/').split('/')[-1][:-4] + '.png',img)

# print('aaaa')





############# 혼자서 데이터 확인할때 해봄##################################

# img = cv2.imread('container_dataset/train_images/OBJ00013_PS3_K3_NIA0078.png')
# with open('container_dataset/train_labels/OBJ00013_PS3_K3_NIA0078.json', 'r') as file:
#     data = json.load(file)
#
#
# pos_8 = list(map(int,(map(float, data['features'][0]['properties']['object_imcoords'].split(',')))))
# pos_8 = np.array([[pos_8[0:2]],[pos_8[2:4]],[pos_8[4:6]],[pos_8[6:8]]],np.int32)
# img = cv2.polylines(img,[pos_8], True, (255,255,0),2)
#
# cv2.imshow('aa',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
#
# print('aa')
