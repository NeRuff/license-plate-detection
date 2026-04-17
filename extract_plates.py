import csv
from collections import Counter

with open('data/detected_plates.csv', 'r') as f:
    reader = csv.DictReader(f)
    plates = [row['plate'] for row in reader]

plate_counts = Counter(plates)
unique_plates = {plate: count for plate, count in plate_counts.items() if count >= 3 and len(plate) >= 4}

print('='*50)
print('УНИКАЛЬНЫЕ НОМЕРА (распознаны 3+ раз):')
print('='*50)
for plate, count in sorted(unique_plates.items(), key=lambda x: x[1], reverse=True):
    print(f'  {plate} -> {count} раз')

print('\n' + '='*50)
print(f'ВСЕГО уникальных номеров: {len(unique_plates)}')
print('='*50)

with open('data/final_plates.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['plate', 'count'])
    for plate, count in sorted(unique_plates.items(), key=lambda x: x[1], reverse=True):
        writer.writerow([plate, count])

print('\nРезультат сохранен в data/final_plates.csv')
