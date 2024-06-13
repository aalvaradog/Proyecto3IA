import os
import shutil
from sklearn.model_selection import train_test_split

def split_dataset(input_dir, output_dir_70, output_dir_30, split_ratio):
    # Crear directorios para el dataset dividido
    os.makedirs(output_dir_70, exist_ok=True)
    os.makedirs(output_dir_30, exist_ok=True)
    
    for category in ['train', 'test', 'valid']:
        category_dir = os.path.join(input_dir, category)
        output_category_dir_70 = os.path.join(output_dir_70, category)
        output_category_dir_30 = os.path.join(output_dir_30, category)
        
        os.makedirs(output_category_dir_70, exist_ok=True)
        os.makedirs(output_category_dir_30, exist_ok=True)
        
        for folder in os.listdir(category_dir):
            folder_path = os.path.join(category_dir, folder)
            if os.path.isdir(folder_path):
                files = os.listdir(folder_path)
                train_files, val_files = train_test_split(files, test_size=split_ratio, random_state=42)
                
                train_folder = os.path.join(output_category_dir_70, folder)
                val_folder = os.path.join(output_category_dir_30, folder)
                
                os.makedirs(train_folder, exist_ok=True)
                os.makedirs(val_folder, exist_ok=True)
                
                for file in train_files:
                    shutil.copy(os.path.join(folder_path, file), os.path.join(train_folder, file))
                    
                for file in val_files:
                    shutil.copy(os.path.join(folder_path, file), os.path.join(val_folder, file))

input_directory = r'C:\Users\tian_\Desktop\GitHub\Proyecto3IA\Data'
output_directory_70 = r'C:\Users\tian_\Desktop\GitHub\Proyecto3IA\DataSplit\Data7030\70'
output_directory_30 = r'C:\Users\tian_\Desktop\GitHub\Proyecto3IA\DataSplit\Data7030\30'
split_dataset(input_directory, output_directory_70, output_directory_30, 0.3)

output_directory_90 = r'C:\Users\tian_\Desktop\GitHub\Proyecto3IA\DataSplit\Data9010\90'
output_directory_10 = r'C:\Users\tian_\Desktop\GitHub\Proyecto3IA\DataSplit\Data9010\10'
split_dataset(input_directory, output_directory_90, output_directory_10, 0.1)


