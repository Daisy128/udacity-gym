import os
import csv

def extract_timestamp_from_filename(filename):
    # 从文件名中提取最后三部分的时间戳，如 '36_07_598'
    return '_'.join(filename.split('_')[-3:])

def is_within_timestamp_range(filename, start_timestamp, end_timestamp):
    file_timestamp = extract_timestamp_from_filename(filename)
    return start_timestamp <= file_timestamp <= end_timestamp

def delete_images_in_range(image_folder, start_timestamp, end_timestamp):
    for image in os.listdir(image_folder):
        if is_within_timestamp_range(image, start_timestamp, end_timestamp):
            image_path = os.path.join(image_folder, image)
            os.remove(image_path)
            print(f"Deleted file: {image_path}")

def delete_lines_in_range(csv_path, start_timestamp, end_timestamp):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows_to_keep = []
        for row in reader:
            row_timestamp = extract_timestamp_from_filename(row[0])  # 假设时间戳在第一列
            if not (start_timestamp <= row_timestamp <= end_timestamp):
                rows_to_keep.append(row)

    # 将过滤后的行重新写回CSV文件
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows_to_keep)

    print(f"Updated CSV file: {csv_path}")

# 主要执行部分
data_folder = '/home/jiaqq/Documents/ase22/datasets/dataset1'
track = 'track3_sim2'
data_type = 'normal'
excel_path = 'driving_log.csv'

# 输入两个时间戳的最后三位
start_timestamp = '45_20_461'
end_timestamp = '45_22_801'

image_folder = os.path.join(data_folder, track, data_type, 'IMG')
print("Images come from: " + image_folder)
delete_images_in_range(image_folder, start_timestamp, end_timestamp)

csv_path = os.path.join(data_folder, track, data_type, excel_path)
delete_lines_in_range(csv_path, start_timestamp, end_timestamp)
