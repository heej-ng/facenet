import os

def rename_images(folder_path):
    # 주어진 폴더 경로에서 모든 하위 폴더를 찾음
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        # 하위 폴더 이름에서 personid를 추출
        person_id = os.path.basename(subfolder)

        # 하위 폴더 내의 이미지 파일들을 찾음
        image_files = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

        # 이미지 파일들을 personid_seq 형태로 이름 변경
        for i, image_file in enumerate(image_files, start=1):
            _, ext = os.path.splitext(image_file)
            new_name = f"{person_id}_{i}{ext}"
            new_path = os.path.join(subfolder, new_name)
            
            # 파일 이름 변경
            os.rename(image_file, new_path)
            print(f"Renamed: {image_file} -> {new_path}")

if __name__ == "__main__":
    folder_path = "/Users/dave/Desktop/얼굴사진"  # 실제 폴더 경로로 바꿔주세요.
    rename_images(folder_path)
