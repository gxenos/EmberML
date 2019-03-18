from random import shuffle
import argparse
import os
import ember

#----------Parser----------
parser = argparse.ArgumentParser()
parser.add_argument("original_filepath", help="Original training file location.", type=str)
parser.add_argument("number_of_train_lines", help="Total lines amount.", type=int)
parser.add_argument("original_test_filepath", help="Original test file location.", type=str)
parser.add_argument("number_of_test_lines", help="Total test lines amount.", type=int)
args = parser.parse_args()


#----------New Balanced Training Set----------
original_filepath = args.original_filepath

f = open(original_filepath, "r")
lines = f.readlines()
f.close()

counter = 0

benign_counter = 0
malware_counter = 0

train_filepath = 'train.jsonl'
train = open(train_filepath, "w")

train_list = []

for line in lines:
    if "\"label\": 0" in line or "\"label\": 1" in line:
        train_list.append(line)

print(f"{'-'*10} Training File Info {'-'*10}")
print(f"Total Train original file Benign & Malware lines: {len(train_list)}")
lines = train_list
shuffle(lines)

for line in lines:
    if "\"label\": 0" in line:
        if benign_counter < args.number_of_train_lines / 2:
            train.write(line)
            benign_counter += 1
    else:
        if malware_counter < args.number_of_train_lines / 2:
            train.write(line)
            malware_counter += 1
train.close()
print(f"New Train File lines:\nBenign: {benign_counter} Malware: {malware_counter}")

#----------New Balanced Test Set----------
original_test_filepath = args.original_test_filepath
new_test_filepath = "test.jsonl"

f = open(original_test_filepath, "r")
lines = f.readlines()
f.close()
shuffle(lines)
print(f"{'-'*10} Test File Info {'-'*10}")
print(f"Total Test original file Benign & Malware lines: {len(lines)}")

benign_counter = 0
malware_counter = 0
test = open(new_test_filepath, "w")
for line in lines:
    if "\"label\": 0" in line:
        if benign_counter < args.number_of_test_lines / 2:
            test.write(line)
            benign_counter += 1
    else:
        if malware_counter < args.number_of_test_lines / 2:
            test.write(line)
            malware_counter += 1
print(f"New Test File lines:\nBenign: {benign_counter} Malware: {malware_counter}")
test.close()

# ---------- Vectorization ----------

def create_custom_vectorized_features(data_dir, train_file_name, test_file_name, train_rows, test_rows):
    """
    Create feature vectors from raw features and write them to disk
    """
    print("Vectorizing training set")
    X_path = os.path.join(data_dir, "X_train.dat")
    y_path = os.path.join(data_dir, "y_train.dat")
    raw_feature_paths = [os.path.join(data_dir, train_file_name)]
    ember.vectorize_subset(X_path, y_path, raw_feature_paths, train_rows)

    print("Vectorizing test set")
    X_path = os.path.join(data_dir, "X_test.dat")
    y_path = os.path.join(data_dir, "y_test.dat")
    raw_feature_paths = [os.path.join(data_dir, test_file_name)]
    ember.vectorize_subset(X_path, y_path, raw_feature_paths, test_rows)


print(f"{'-'*10} Vectorization Info {'-'*10}")
create_custom_vectorized_features('.', 'train.jsonl', 'test.jsonl', args.number_of_train_lines, args.number_of_test_lines)
