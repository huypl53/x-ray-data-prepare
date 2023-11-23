
def parse_label_file(file_path: str): 
    label_string = open(file_path).read()
    label_string_lines = label_string.split('\n')
    
    labels = [ [float(v) for v in line.split()] for line in label_string_lines]
    for label in labels:
        if not len(label): 
            continue
        label[0] = int(label[0])

    return labels

def label_num2str(labels):
    label_lines = [ ' '.join( [str(v) for v in label] ) for label in labels]
    label_str = '\n'.join(label_lines)
    return label_str

def save_labels(labels, file_path):
    label_str = label_num2str(labels)
    with open(file_path, 'w') as fw:
        fw.write(label_str)
