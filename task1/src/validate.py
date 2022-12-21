with open('/home/group14/CS433FinalProject/task1/target/output/oracle_predictions.txt') as f:
    oracle_predictions = f.readlines()
files = [line.split()[0] for line in oracle_predictions]
labels = [line.split()[2] for line in oracle_predictions]

with open('/home/group14/CS433FinalProject/task1/target/output/predictions.txt') as f:
    predictions = f.readlines()
pred_files = [line.split()[0] for line in predictions]
predictions = [line.split()[1] for line in predictions]

accuracy, total = 0, len(labels)
error_list = []
for i in range(total):
    if files[i] != pred_files[i]:
        print(f'Error: {i} {files[i]} != {pred_files[i]}')
        break
    if labels[i] == predictions[i]:
        accuracy += 1
    else:
        error_list.append([i + 1, files[i], labels[i], predictions[i]])
print(f'Accuracy: {accuracy/total:.2%} ({accuracy}/{total})')

with open('/home/group14/CS433FinalProject/task1/target/output/error_file_list.txt', 'w') as f:
    for i in error_list:
        # print(i)
        f.write(f'\'{i[1]}\',\n')