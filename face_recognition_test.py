# facenet을 이용해 데이터셋 내 각 얼굴에 대한 얼굴 임베딩 계산
from numpy import expand_dims
from numpy import savez_compressed
import keras
from keras.models import load_model
from os import listdir
from os.path import isdir
from PIL import Image
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from numpy import load
from sklearn.preprocessing import Normalizer
import pickle
import tensorflow as tf
from numpy import load
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import boto3
resource_s3 = boto3.resource('s3')




from sklearn.cluster import KMeans


def extract_face(filename, required_size=(160, 160)):
	# 파일에서 이미지 불러오기
	image = Image.open(filename)
	# RGB로 변환, 필요시
	image = image.convert('RGB')
	# 배열로 변환
	pixels = asarray(image)
	# 감지기 생성, 기본 가중치 이용
	detector = MTCNN()
	# 이미지에서 얼굴 감지
	results = detector.detect_faces(pixels)
	#0번째 얼굴에서 경계 상자 추출

	x1, y1, width, height = results[0]['box']
	# 버그 수정
	x1, y1 = abs(x1), abs(y1)
	x2, y2 = x1 + width, y1 + height
	# 얼굴 추출
	face = pixels[y1:y2, x1:x2]
  # 모델 사이즈로 픽셀 재조정
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = asarray(image)
	return face_array


# 디렉토리 안의 모든 이미지를 불러오고 이미지에서 얼굴 추출
def load_faces(directory):
	faces = list()
	# 파일 열거
	for filename in listdir(directory):
		# 경로
		path = directory + filename
		# 얼굴 추출
		face = extract_face(path)
		# 저장
		faces.append(face)
	return faces

# 이미지를 포함하는 각 클래스에 대해 하나의 하위 디렉토리가 포함된 데이터셋을 불러오기
def load_dataset(directory):
	X, y = list(), list()
	# 클래스별로 폴더 열거
	for subdir in listdir(directory):
		# 경로
		path = directory + subdir + '/'
		# 디렉토리에 있을 수 있는 파일을 건너뛰기(디렉토리가 아닌 파일)
		if not isdir(path):
			continue
		# 하위 디렉토리의 모든 얼굴 불러오기
		faces = load_faces(path)
		# 레이블 생성
		labels = [subdir for _ in range(len(faces))]
		# 진행 상황 요약
		print('>%d개의 예제를 불러왔습니다. 클래스명: %s' % (len(faces), subdir))
		# 저장
		X.extend(faces)
		y.extend(labels)
	return asarray(X), asarray(y)



# 하나의 얼굴의 얼굴 임베딩 얻기
def get_embedding(main_model, face_pixels):
	face_pixels = face_pixels.astype('int32')
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	samples = expand_dims(face_pixels, axis=0)
	yhat = main_model.predict(samples)
	return yhat[0]

def return_score(main_model, main_face_model):
	print(type(main_model))
	testX, testy = load_dataset(resource_s3.Object('capstonefaceimg','load/data/5-celebrity-faces-dataset/val/'))
	savez_compressed('G:/내 드라이브/capstone_2/data/5-celebrity-faces-dataset_test.npz', testX, testy)


	data_test = load(resource_s3.Object('capstonefaceimg','load/data/5-celebrity-faces-dataset_test.npz'))
	testX, testy = data_test['arr_0'], data_test['arr_1']
	newTestX = list()


	for face_pixels in testX:
		embedding = get_embedding(main_model, face_pixels)
		newTestX.append(embedding)
	newTestX = asarray(newTestX)

	# 배열을 하나의 압축 포맷 파일로 저장
	savez_compressed('5-celebrity-faces-embeddings_test.npz', newTestX, testy)

	data_test = load(s3.Object('capstonefaceimg','load/data/5-celebrity-faces-embeddings_test.npz'))

	testX, testy = data_test['arr_0'], data_test['arr_1']
	# 입력 벡터 일반화
	in_encoder = Normalizer(norm='l2')
	testX = in_encoder.transform(testX)
	#face_model = pickle.load(open('G:/내 드라이브/capstone_2/finalized_model.h5', 'rb'))

	random_face_emb = testX[0]
	random_face_class = testy[0]

	samples = expand_dims(random_face_emb, axis=0)
	yhat_class = main_face_model.predict(samples)
	yhat_prob = main_face_model.predict_proba(samples)
	class_index = yhat_class[0]
	print(str(class_index) + "---> cmd")
	return(class_index)

#print(f"tensorflow version: {tf.__version__}")
#print(f"keras version: {tf.keras.__version__}")
#return_score()
#model = tf.keras.models.load_model("G:/내 드라이브/capstone_2/data/facenet_keras.h5")
# 테스트 셋에서 각 얼굴을 임베딩으로 변환하기
