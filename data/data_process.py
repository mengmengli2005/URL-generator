import random

# 提取多个phishTank文件里的malicious URL
def process_phishTank(file_phishTank_list, No_bad, file_name):
    fout = open(file_name,"w")
    fout.truncate()
    badlist = []
    with open(file_phishTank_list) as file:
        for line in file:
            with open(line.strip()) as subfile:
                for r in subfile:
                    tmp = r.split(',')[1].strip()
                    if tmp == 'url':
                        continue
                    record = "http://" + tmp.split('//')[1] + '\n'
                    badlist.append(record)

    sample = random.sample(badlist, No_bad)
    print('#bad : %d', len(sample))

    for line in sample:
        fout.write(line)

    print('#all bad: ', len(badlist))
    print('#bad : ', len(sample))
    fout.close()

def process_kaggle_randomize(original_file, No_good, output_file):
    fout = open(output_file, "w")
    fout.truncate()
    with open(original_file, encoding = "ISO-8859-1") as file:
        goodlist = []

        for line in file:
            # print(line)
            record = line.split(',')
            url = record[0]
            lable = record[1].strip()
            if url == 'url' or (lable != 'good' and lable != 'bad'):
                continue
            if lable == 'good':
                out = 'http://' + url.strip() + '\n'
                goodlist.append(out)

        sample_good = random.sample(goodlist, No_good)
        print('#good : %d', len(sample_good))

        for line in sample_good:
            fout.write(line)

        print('#all good: ', len(goodlist))
        print('#good : ', len(sample_good))

    fout.close()





process_phishTank("phishTank_list.txt", 10000, "PhishTank_bad_10T.txt")
process_kaggle_randomize("Kaggle_data.txt", 10000, "Kaggle_good_10T.txt")