import os


def split_files(file_path, out_dir):
    num_lines = 100
    file_num = 0
    buffer = None
    print("Source file ", file_path)
    for i, l in enumerate(open(file_path, 'r')):
        if i % num_lines == 0:
            if buffer:
                buffer.close()
            file_num += 1
            fname = os.path.join(out_dir, 'split' + str(file_num) + '.txt')
            print("writing split file ", fname)
            buffer = open(fname, 'w+')
        buffer.write(l)
    buffer.close()


split_files("data/complete/shakespere.txt", "data/train_files")
