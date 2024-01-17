import csv

input_file="C:/Users/22279/Desktop/数据集/风机功率数据/wtbdata_245days.csv"
output_path="C:/Users/22279/Desktop/数据集/风机功率数据/"
column_name="TurbID"

output_writers={}

with open(input_file,'r') as file:
    reader=csv.DictReader(file)

    for row in reader:
        value=row[column_name]

        if value not in output_writers:
            output_file=output_path+"turb_"+str(value)+".csv"
            output=open(output_file,'w',newline='')
            writer=csv.DictWriter(output,fieldnames=reader.fieldnames)
            writer.writeheader()
            output_writers[value]=writer

        output_writers[value].writerow(row)

for writer in output_writers.values():
    writer.close()
