import csv
import re
import os
import sys

if len(sys.argv) < 2:
    print("CSV_DIR ~> directory of your unnormalized csv")
else:
    csv_dir = sys.argv[1]
    for root, dirs, files in os.walk(csv_dir, topdown=False):
        for file in files:
            name, ext = file.split('.')
            if ext == 'csv':
                with open(os.path.join(csv_dir, file), 'r') as csvfile:
                    csvreader = csv.reader(csvfile, delimiter=',')
                    csv_out = open(os.path.join(csv_dir, name + '-normalized.csv'), 'w')
                    csvwriter = csv.writer(csv_out)
                    result = []
                    for row in csvreader:
                        if (len(row) > 0):
                            if not re.match("^\d+?\.\d+?$", row[0]) is None:
                                if len(result) == 0:
                                    result.append(row[0])
                                    csvwriter.writerow(row)
                                elif result[len(result)-1] != row[0]:
                                    result.append(row[0])
                                    csvwriter.writerow(row)
