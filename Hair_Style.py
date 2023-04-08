import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# MUNIT 모델 로드
munit = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# 웹캠 초기화
cap = cv2.VideoCapture(0)

while True:
    # 웹캠으로부터 프레임 받아오기
    ret, frame = cap.read()
    if not ret:
        break
    
    # 프레임을 모델에 입력할 수 있도록 전처리
    content_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    content_image = tf.image.resize(content_image, (256, 256))
    content_image = tf.expand_dims(content_image, axis=0)
    content_image = content_image / 255.0
    
    # 스타일 이미지를 불러와 전처리
    style_image = cv2.imread('style.jpg')
    style_image = cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB)
    style_image = tf.image.resize(style_image, (256, 256))
    style_image = tf.expand_dims(style_image, axis=0)
    style_image = style_image / 255.0
    
    # MUNIT 모델에 입력하고 결과 출력
    stylized_image = munit(content_image, style_image)[0]
    stylized_image = np.array(stylized_image)[0]
    stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)
    
    # 출력
    cv2.imshow('Webcam', stylized_image)
    
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) == 27:
        break

# 종료
cap.release()
cv2.destroyAllWindows()
