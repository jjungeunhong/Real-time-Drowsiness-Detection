import cv2
import tensorflow
import numpy as np
import beepy

def preprocessing(frame):
	size = (224, 224)
	frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)

	frame_normalized = (frame_resized.astype(np.float32) / 127.0) - 1
	frame_reshaped = frame_normalized.reshape((1, 224, 224, 3))

	return frame_reshaped

def beepsound():
	beepy.beep(sound=7)

model_filename = 'keras_model.h5'
model = tensorflow.keras.models.load_model(model_filename)

capture = cv2.VideoCapture(1)

capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

sleep_cnt = 1
while True:
	ret, frame = capture.read()
	if ret == True:
		print("Read Success!")
	
	frame_flipped = cv2.flip(frame, 1)

	cv2.imshow("VideoFrame", frame_flipped)

	if cv2.waitKey(200) > 0:
		break

	preprocessed = preprocessing(frame_flipped)
	prediction = model.predict(preprocessed)

	if prediction[0, 0] < prediction[0, 1]:
		print("You\'re falling asleep!")
		sleep_cnt += 1

		if sleep_cnt % 30 == 0:
			sleep_cnt = 1
			print("You\'re sleeping for 30 seconds!")
			beepsound()
			break
	else:
		print("You\'re awake")
		sleep_cnt = 1

capture.release()
cv2.destroyAllWindows()