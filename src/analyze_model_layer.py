import h5py
import tensorflow as tf

# HDF5 파일 경로 (모델이 저장된 파일)
date = '20231106-205045'
# fine_tuned_model_path = f'/Users/dave/.deepface/weights/facenet512_weights_custom_39-re-layer.h5'
original_file_path = f'/Users/dave/.deepface/weights/facenet512_weights_original.h5'
fine_tuned_model_path = f'/Users/dave/Desktop/facenet-model/20180402-114759/facenet512_weights_275.h5'

def read_model_layer(h5_file_path):
  # HDF5 파일을 읽어옵니다.
  with h5py.File(h5_file_path, 'r') as h5_file:
      # 모델의 레이어 정보를 출력합니다.
      def print_hdf5_item(name, obj):
          if isinstance(obj, h5py.Group):
              print(f"Group: {name}")
          elif isinstance(obj, h5py.Dataset):
              print(f"Dataset: {name}")
              # 모델 레이어 이름 출력
              if 'model_weights' in name:
                  print(f"Layer Name: {name}")
      
      # HDF5 파일 내의 모든 요소를 반복하면서 출력
      h5_file.visititems(print_hdf5_item)


def compare_model_architectures(base_model_path, fine_tuned_model_path):
    # HDF5 파일에서 모델 구조 및 레이어를 읽어옵니다.
    base_model = h5py.File(base_model_path, 'r')
    fine_tuned_model = h5py.File(fine_tuned_model_path, 'r')

    base_layers = []
    fine_tuned_layers = []

    # 기존 모델의 레이어 이름을 추출합니다.
    def extract_layers(name, obj):
        if isinstance(obj, h5py.Group):
            base_layers.append(name)

    base_model.visititems(extract_layers)

    # 파인튜닝 모델의 레이어 이름을 추출합니다.
    def extract_layers_fine_tuned(name, obj):
        if isinstance(obj, h5py.Group):
            fine_tuned_layers.append(name)

    fine_tuned_model.visititems(extract_layers_fine_tuned)

    # 레이어 개수와 이름을 비교합니다.
    if len(base_layers) != len(fine_tuned_layers):
        print("레이어 개수가 일치하지 않습니다. 호환되지 않는 모델 아키텍처입니다.")
        print(f"기존 모델 레이어 개수: {len(base_layers)}")
        print(f"파인튜닝 모델 레이어 개수: {len(fine_tuned_layers)}")
        # return

    for i in range(max(len(base_layers), len(fine_tuned_layers))):
        if i >= len(base_layers):
            print(f"파인튜닝 모델에 추가된 레이어: {fine_tuned_layers[i]}")
        elif i >= len(fine_tuned_layers):
            print(f"기존 모델에 추가된 레이어: {base_layers[i]}")

    for base_layer, fine_tuned_layer in zip(base_layers, fine_tuned_layers):
        if base_layer != fine_tuned_layer:
            print("레이어 이름이 일치하지 않습니다. 호환되지 않는 모델 아키텍처입니다.")
            print(f"기존 모델 레이어 이름: {base_layer}")
            print(f"파인튜닝 모델 레이어 이름: {fine_tuned_layer}")
            # return

    # 모델 아키텍처가 일치하면 이제 가중치 변수를 확인할 수 있습니다.
    for base_layer, fine_tuned_layer in zip(base_layers, fine_tuned_layers):
        base_weights = base_model[base_layer].get('kernel:0')
        fine_tuned_weights = fine_tuned_model[fine_tuned_layer].get('kernel:0')

        # 가중치 변수 형태를 비교합니다.
        if base_weights.shape != fine_tuned_weights.shape:
            print(f"레이어 '{base_layer}'와 '{fine_tuned_layer}'의 가중치 변수 형태가 다릅니다.")
            print(f"기존 모델 가중치 변수 형태: {base_weights.shape}")
            print(f"파인튜닝 모델 가중치 변수 형태: {fine_tuned_weights.shape}")
            # return

    # 모든 레이어와 가중치 변수가 호환되면 아키텍처가 일치합니다.
    print("모델 아키텍처가 호환됩니다.")

def find_layer_info(model_file_path):
  # H5 모델 파일 경로 설정
  
  # H5 파일 열기
  model_h5 = h5py.File(model_file_path, 'r')
  
  # 'top_level_model_weights' 레이어의 정보 출력
  if 'top_level_model_weights' in model_h5:
      top_level_weights = model_h5['top_level_model_weights']
      
      # 'top_level_model_weights' 레이어의 가중치 출력
      for layer_name in top_level_weights.keys():
          layer_weights = top_level_weights[layer_name]
          print(f'Layer Name: {layer_name}')
          for weight_name in layer_weights.keys():
              weight_value = layer_weights[weight_name][()]
              print(f'  Weight Name: {weight_name}')
              print(f'  Weight Value: {weight_value}')
  else:
      print("'top_level_model_weights' 레이어가 모델에 존재하지 않습니다.")
  
  # H5 파일 닫기
  model_h5.close()

def print_weights(model_file_path):
  # 모델 로드
  model = tf.keras.models.load_model(model_file_path)

  # 'Logits' 레이어의 가중치 출력
  if 'Logits' in model.layers:
      logits_layer = model.get_layer('Logits')
      logits_weights = logits_layer.get_weights()

      for i, weights in enumerate(logits_weights):
          print(f'Logits Layer Weights {i}:')
          print(weights)
  else:
      print("'Logits' 레이어를 찾을 수 없습니다.")

  # 모델 요약 정보 출력
  model.summary()

# read_model_layer(fine_tuned_model_path)
# 두 모델의 HDF5 파일 경로를 지정하여 함수를 호출합니다.
compare_model_architectures(original_file_path, fine_tuned_model_path)
# find_layer_info(fine_tuned_model_path)
# print_weights(fine_tuned_model_path)