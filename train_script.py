import sys, json, requests
import pandas as pd

class FileFormatError(Exception):
    def __init__(self, f, *args):
        super().__init__(args)
        self.fname = f

    def __str__(self):
        return f'The file "{self.fname}" does not have a valid format! Please review guide and double check your file.'
    
class DatasetExistsError(Exception):
    def __init__(self, d, *args):
        super().__init__(args)
        self.dname = d

    def __str__(self):
        return f'The name of the uploaded dataset "{self.dname}" already exists! Please use a different name and try again.'
    
class FeatureLabelMismatchError(Exception):
    def __init__(self, f, l, *args):
        super().__init__(args)
        self.fname = f
        self.lname = l

    def __str__(self):
        return f'The file "{self.fname}" and file "{self.lname}" have different number of rows or columns! Please review guide and double check your files.'

def read_in():
    lines = sys.stdin.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
    return lines

if __name__ == '__main__':
    print('Running Python Script for Custom Training Dataset Upload Verification')
    line = read_in()

    with open('./public/custom.json', 'r') as json_file:
        json_str = json_file.read()
        json_body = json.loads(json_str)
        if line[3] in json_body.keys():
            raise DatasetExistsError(line[3])

    feature_df = None
    try:
        feature_df = pd.read_csv(line[0])
    except:
        try:
            feature_df = pd.read_excel(line[0])
        except:
            f_name = line[0].split('/')[-1]
            raise FileFormatError(f_name)
    
    if feature_df is None:
        f_name = line[0].split('/')[-1]
        raise FileFormatError(f_name)

    label_df = None
    try:
        label_df = pd.read_csv(line[1])
    except:
        try:
            label_df = pd.read_excel(line[1])
        except:
            f_name = line[1].split('/')[-1]
            raise FileFormatError(f_name)
        
    clean_df = None
    try:
        clean_df = pd.read_csv(line[2])
    except:
        try:
            clean_df = pd.read_excel(line[2])
        except:
            f_name = line[2].split('/')[-1]
            raise FileFormatError(f_name)
        
    if len(label_df.columns) != 1:
        f_name = line[1].split('/')[-1]
        raise FileFormatError(f_name)
    
    if len(feature_df) != len(label_df) or len(feature_df) != len(clean_df) or len(feature_df.columns) != len(clean_df.columns):
        f_name = line[0].split('/')[-1]
        l_name = line[1].split('/')[-1]
        raise FeatureLabelMismatchError(f_name, l_name)

    with open('./public/custom.json', 'r') as json_file:
        json_str = json_file.read()
        json_body = json.loads(json_str)
        # print(json_body)
        json_body.update({line[3]: [line[0].split('/')[-1], line[1].split('/')[-1], line[2].split('/')[-1]]})
        new_json = json.dumps(json_body)

    with open('./public/custom.json', 'w') as json_file:
        json_file.write(new_json)

    desc_filename = "./dataset/" + line[3] + "/" + line[3] + "_description.txt"
    with open(desc_filename, "w") as desc_file:
        desc_file.write(line[4])