import sys, json, requests
import pandas as pd

class FileFormatError(Exception):
    def __init__(self, f, *args):
        super().__init__(args)
        self.fname = f

    def __str__(self):
        return f'The file "{self.fname}" does not have a valid format! Please review guide and double check your file. If the file is a training dataset, contact the administrator for assistance.'
    
class DatasetExistsError(Exception):
    def __init__(self, d, *args):
        super().__init__(args)
        self.dname = d

    def __str__(self):
        return f'The name of the uploaded testing dataset "{self.dname}" already exists! Please use a different name and try again.'
    
class FeatureLabelMismatchError(Exception):
    def __init__(self, f, l, *args):
        super().__init__(args)
        self.fname = f
        self.lname = l

    def __str__(self):
        return f'The feature file "{self.fname}" and label file "{self.lname}" have different number of rows! Please review guide and double check your files.'
    
class TrainTestMismatchError(Exception):
    def __init__(self, f1, f2, *args):
        super().__init__(args)
        self.f1name = f1
        self.f2name = f2

    def __str__(self):
        return f'The feature files "{self.f1name}" and "{self.f2name}" do not have the same number of columns! Please review guide and double check your files.'
    
class TrainingSetNotFoundError(Exception):
    def __init__(self, d, *args):
        super().__init__(args)
        self.dname = d

    def __str__(self):
        return f'The training dataset "{self.dname}" does not exist! Please upload the training dataset first or try again with the correct training dataset name.'

def read_in():
    lines = sys.stdin.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    return lines

if __name__ == '__main__':
    print('Running Python Script for Custom Testing Dataset Upload Verification')
    line = read_in()

    with open('./public/custom.json', 'r') as json_file:
        json_str = json_file.read()
        json_body = json.loads(json_str)
        if line[2] not in json_body.keys():
            raise TrainingSetNotFoundError(line[2])
        
    train_feature = None
    try:
        train_feature = pd.read_csv('./dataset/' + line[2] + '/' + json_body[line[2]][0])
    except:
        try:
            train_feature = pd.read_excel('./dataset/' + line[2] + '/' + json_body[line[2]][0])
        except:
            f_name = json_body[line[2]][0].split('/')[-1]
            raise FileFormatError(f_name)

    feature_df = None
    try:
        feature_df = pd.read_csv(line[0])
    except:
        try:
            feature_df = pd.read_excel(line[0])
        except:
            f_name = line[0].split('/')[-1]
            raise FileFormatError(f_name)
        
    if train_feature is None:
        f_name = json_body[line[2]][0]
        raise FileFormatError(f_name)
    
    if feature_df is None:
        f_name = line[0].split('/')[-1]
        raise FileFormatError(f_name)
    
    if len(feature_df.columns) != len(train_feature.columns):
        f_name = line[0].split('/')[-1]
        raise TrainTestMismatchError(f_name, json_body[line[2]][0])

    label_df = None
    try:
        label_df = pd.read_csv(line[1])
    except:
        try:
            label_df = pd.read_excel(line[1])
        except:
            f_name = line[1].split('/')[-1]
            raise FileFormatError(f_name)
        
    if len(label_df.columns) != 1:
        f_name = line[1].split('/')[-1]
        raise FileFormatError(f_name)
    
    if len(feature_df) != len(label_df):
        f_name = line[0].split('/')[-1]
        l_name = line[1].split('/')[-1]
        raise FeatureLabelMismatchError(f_name, l_name)

    with open('./public/custom_test.json', 'r') as json_file:
        json_str = json_file.read()
        json_body = json.loads(json_str)
        # print(json_body)
        if line[2] not in json_body.keys():
            json_body[line[2]] = [[line[0].split('/')[-1], line[1].split('/')[-1]]]
        else:
            json_body[line[2]].append([line[0].split('/')[-1], line[1].split('/')[-1]])

    with open('./public/custom_test.json', 'w') as json_file:
        json_file.write(json.dumps(json_body))