import os
import csv

def delete_image(image_folder, timestamp):
    for image in os.listdir(image_folder):
        if timestamp in image:
            image_path = os.path.join(image_folder, image)
            os.remove(image_path)
            print(f"Deleted file: {image_path}")

def delete_lines(csv_path, timestamp):
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows_to_keep = [row for row in reader if timestamp not in ','.join(row)]

    # Write the filtered rows back to the CSV file
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows_to_keep)

    print(f"Updated CSV file: {csv_path}")

data_folder = '/home/jiaqq/Documents/ase22/datasets/dataset1'
track = 'track3_sim2'
data_type = 'normal'
excel_path = 'driving_log.csv'

timestamp = '16_10_39_838'

image_folder = os.path.join(data_folder, track, data_type, 'IMG')
print("images come from: " + image_folder)
delete_image(image_folder, timestamp)

csv_path = os.path.join(data_folder, track, data_type, excel_path)
delete_lines(csv_path, timestamp)
