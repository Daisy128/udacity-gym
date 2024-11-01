import os
import csv

def extract_ts(filename):
    # 从文件名中提取最后三部分的时间戳，如 '36_07_598'
    return '_'.join(filename.split('_')[-3:])

def within_range(filename, start_ts, end_ts):
    file_ts = extract_ts(filename)
    return start_ts <= file_ts <= end_ts

def delete_images_in_range(image_folder, start_ts, end_ts):
    for image in os.listdir(image_folder):
        if within_range(image, start_ts, end_ts):
            image_path = os.path.join(image_folder, image)
            os.remove(image_path)
            print(f"Deleted file: {image_path}")

def delete_lines_in_range(csv_path, start_ts, end_ts):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows_to_keep = []
        for row in reader:
            row_ts = extract_ts(row[0]) #只找第一列的timestamp，删除整行
            if not (start_ts <= row_ts <= end_ts):
                rows_to_keep.append(row)

    #过滤后的行重新写回CSV
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows_to_keep)

    print(f"Updated CSV file: {csv_path}")

data_folder = '/home/jiaqq/Documents/ase22/datasets/dataset1'
track = 'track3_sim2'
data_type = 'normal'
excel_path = 'driving_log.csv'

# 两个时间戳的最后三位
start_ts = '45_20_461'
end_ts = '45_22_801'

image_folder = os.path.join(data_folder, track, data_type, 'IMG')
print("Images come from: " + image_folder)
delete_images_in_range(image_folder, start_ts, end_ts)

csv_path = os.path.join(data_folder, track, data_type, excel_path)
delete_lines_in_range(csv_path, start_ts, end_ts)
