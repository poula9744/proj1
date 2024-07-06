from fastapi import FastAPI, File, UploadFile

# STEP 1
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.components import processors
from mediapipe.tasks.python import vision

# STEP 2: 추론기 미리 한 번만 만들어 두고 계속 쓴다
base_options = python.BaseOptions(model_asset_path='models\cls\efficientnet_lite0.tflite') #추론기 기본이 되는 옵션
options = vision.ImageClassifierOptions(
    base_options=base_options, max_results=2) #확률값이 높은 상위 1개
classifier = vision.ImageClassifier.create_from_options(options)

app = FastAPI()

from PIL import Image
import numpy as np
import io
@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile):
    content = await file.read() #파일을 주고받는 빈 껍데기
    #content --> jpg 파일인데 http통신에서는 파일이 character type 왔다갔다 함
    #create_from_file는 내장메모리에서 불러오는 것


    # 1. text -> binary
    # 2.binary -> PIL Image
    # STEP 3: Load the input image.
    binary = io.BytesIO(content)
    pil_img = Image.open(binary)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(pil_img))

    # STEP 4: Classify the input image. 추론(손댈일이 없음)
    classification_result = classifier.classify(image)
    # print(classification_result)

    # STEP 5: 사용자에게 어떻게 보여줄 것인가?
    top_category = classification_result.classifications[0].categories[0]
    result = f"{top_category.category_name} ({top_category.score:.2f})"

    return {"result": result}

