import urllib.request
#.request를 안붙이면 아래 자동 다운로드가 오류남
IMAGE_FILENAMES = ['burger.jpg', 'cat.jpg', 'dog.jpg']

# for name in IMAGE_FILENAMES:
#   url = f'https://storage.googleapis.com/mediapipe-tasks/image_classifier/{name}'
#   urllib.request.urlretrieve(url, name)

# https://storage.googleapis.com/mediapipe-tasks/image_classifier/burger.jpg
# https://storage.googleapis.com/mediapipe-tasks/image_classifier/cat.jpg

# import cv2
# import math

# DESIRED_HEIGHT = 480
# DESIRED_WIDTH = 480

# def resize_and_show(image):
#   h, w = image.shape[:2]
#   if h < w:
#     img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
#   else:
#     img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
# #   cv2_imshow(img) 
#   cv2.imshow("test", img)   #이미지 파일을 열겠다
#   cv2.waitKey(0)  #키가 눌릴 때까지 기다리겠다


# # Preview the images.

# images = {name: cv2.imread(name) for name in IMAGE_FILENAMES}
# for name, image in images.items():
#   print(name)
#   resize_and_show(image)





# STEP 1: Import the necessary modules. 패키지를 가져온다
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: Create an ImageClassifier object. 추론기 객체 만들기, 추론기마다 옵션이 다름
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite') #추론기 기본이 되는 옵션
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=2) #확률값이 높은 상위 1개
classifier = vision.ImageClassifier.create_from_options(options)


# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILENAMES[2])

# STEP 4: Classify the input image. 추론(손댈일이 없음)
classification_result = classifier.classify(image)
# print(classification_result)

# STEP 5: Process the classification result. In this case, visualize it. 사용자에게 어떻게 보여줄 것인가?
top_category = classification_result.classifications[0].categories[0]
print(f"{top_category.category_name} ({top_category.score:.2f})") #cheeseburger (0.98)

