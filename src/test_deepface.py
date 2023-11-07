from deepface import DeepFace

def test(p1,p2):
  return DeepFace.verify(img1_path=p1,img2_path=p2,model_name="Facenet512", detector_backend='retinaface',distance_metric='euclidean')

dir = "/Users/dave/Desktop/얼굴사진/"
print(test(dir+'dave1.jpg',dir+'song1.jpeg'))