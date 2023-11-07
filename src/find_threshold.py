import deepface_test

result_list = deepface_test.deepface_result

def find_statistics():
  # Extract second values from the tuple
  second_values = [x[1] for x in result_list]

  # Calculate minimum and maximum integer values of second values for the range
  min_val = int(min(second_values))
  max_val = int(max(second_values))

  # Iterate over the thresholds
  for th in range(min_val, max_val+1):
    for point in range(0,10):
        threshold = th + point/10
        print(f"--------- Threshold: {threshold} ---------")
        categories = {}  # Using a dictionary to store the statistics per category

        for label, value in result_list:
            if value < threshold:
                categories[label] = categories.get(label, [0, 0])  # [smaller, equal or bigger]
                categories[label][0] += 1
            else:
                categories[label] = categories.get(label, [0, 0])
                categories[label][1] += 1

        rate_sum = 0
        # Print statistics for each label
        for label, counts in categories.items():
            if (label == 'self'):
                rate = counts[0] / (counts[0] + counts[1])
                rate_sum += rate
            elif (label == 'other'):
                rate = counts[1] / (counts[0] + counts[1])
                rate_sum += rate
            print(f"Label: {label}, accuracy rate: {rate:.4f}, smaller cnt: {counts[0]} < Threshold: {threshold} <= equal or bigger cnt: {counts[1]}")
        print(f"Threshold: {threshold}, average accuracy rate: {(rate_sum/2):.4f}")

def find_threshold():
  second_values = [x[1] for x in result_list]
  min_val = int(min(second_values))
  max_val = int(max(second_values))

  best_accuracy = 0
  best_threshold = min_val

  for th in range(min_val, max_val+1):
    for point in range(0,10):
        threshold = th + point/10
        TP, FN, TN, FP = 0, 0, 0, 0     
        for label, value in result_list:
            if label == 'self' and value < threshold:
                TP += 1
            elif label == 'self' and value >= threshold:
                FN += 1
            elif label == 'other' and value < threshold:
                FP += 1
            elif label == 'other' and value >= threshold:
                TN += 1     
        # Calculate accuracy for the given threshold
        accuracy = 0
        if TP + FN > 0:
            accuracy += TP / (TP + FN)
        if FP + TN > 0:
            accuracy += TN / (FP + TN)      
        # Check if this accuracy is the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold      
  print(f"Best Threshold: {best_threshold}, Best Accuracy: {best_accuracy/2}")
find_statistics()
print('---------------------------------')
find_threshold()