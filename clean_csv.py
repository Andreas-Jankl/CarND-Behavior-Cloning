import csv
from os import listdir
from os.path import join

base_dir = 'C:/Users/Zorn/Desktop/Udacity_Self_Driving_Programm/Projects/Project_3_test_2/data/IMG/'

img_files = listdir('data/IMG/')
with open('data/driving_log_clean.csv', 'w',newline='') as wf:
	with open('data/driving_log.csv', 'r') as rf:
		reader = csv.reader(rf)
		writer = csv.writer(wf)
		for row in reader:
			rel_path = row[0].split('/')[-1]
			#print(rel_path)
			if rel_path in img_files:
				#real_path = join(base_dir, rel_path)
				real_path = base_dir+rel_path
				writer.writerow([real_path, row[3]])