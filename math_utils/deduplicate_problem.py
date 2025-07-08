from thefuzz import fuzz
import json
from utils import find_question
from tqdm import tqdm
import sys

def check_duplicate(text1, text2, threshold=90):
    if fuzz.ratio(text1, text2) > threshold:
        return True
    return False

#print(check_duplicate("What is the sum of the first 100 natural numbers?", "What is the sum of the first 100 natural numbers?"))

training_set_path = "data/DeepMath-103K-big-number/DeepMath-103K-big-number_question_everything.jsonl"

training_problem_list = []
with open(training_set_path, "r") as f:
    for line in f:
        data = json.loads(line)
        training_problem_list.append(data)

print("training set size: ", len(training_problem_list))

#test_set_path = "data/combinatorics_test_ge10000.jsonl"
test_set_path = sys.argv[1]
threshold = 90
test_problem_list = []
with open(test_set_path, "r") as f:
    for line in f:
        data = json.loads(line)
        test_problem_list.append(data)

print("test set size: ", len(test_problem_list))

no_duplicate_count = 0
no_duplicate_count_list = []
for problem in tqdm(test_problem_list):
    for training_problem in training_problem_list:
        if check_duplicate(find_question(problem), find_question(training_problem), threshold=threshold):
            print("find duplicate, ratio: ", fuzz.ratio(find_question(problem), find_question(training_problem)))
            print(">>>>>>>>>>>>>>>>>>>>")
            print(find_question(problem))
            print("<<<<<<<<<<<<<<<<<<<<")
            print(find_question(training_problem))
            print(">>>>>>>>>>>>>>>>>>>>")
            break
    else:
        no_duplicate_count += 1
        no_duplicate_count_list.append(problem)

print("test set no duplicate count: ", no_duplicate_count)

with open(test_set_path.replace(".jsonl", "_no_duplicate_threshold_{}.jsonl".format(threshold)), "w") as f:
    for problem in no_duplicate_count_list:
        f.write(json.dumps(problem) + "\n")
