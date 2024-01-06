import json
with open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_data/anno_clean.json', 'r', encoding='utf-8') as file_clean:
    file_clean = json.load(file_clean)
with open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_data/anno_large.json', 'r', encoding='utf-8') as file_large:
    file_large = json.load(file_large)
with open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_clean_source_train.txt', 'w', encoding='utf-8') as raw_clean_src_train,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_clean_source_test.txt', 'w', encoding='utf-8') as raw_clean_src_test,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_clean_source_valid.txt', 'w', encoding='utf-8') as raw_clean_src_valid,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_clean_target_train.txt', 'w', encoding='utf-8') as raw_clean_tgt_train,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_clean_target_test.txt', 'w', encoding='utf-8') as raw_clean_tgt_test,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_clean_target_valid.txt', 'w', encoding='utf-8') as raw_clean_tgt_valid,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_large_source_train.txt', 'w', encoding='utf-8') as raw_large_src_train,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_large_source_test.txt', 'w', encoding='utf-8') as raw_large_src_test,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_large_source_valid.txt', 'w', encoding='utf-8') as raw_large_src_valid,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_large_target_train.txt', 'w', encoding='utf-8') as raw_large_tgt_train,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_large_target_test.txt', 'w', encoding='utf-8') as raw_large_tgt_test,\
     open(r'/gb/HZY/mask-our-model/Fashion-MMT-preprocess/raw_large_target_valid.txt', 'w', encoding='utf-8') as raw_large_tgt_valid:
     for i,j in enumerate(file_clean):
         if j['split'] == 'trn':
            raw_clean_src_train.write(file_clean[i]['en']+"\n")
            raw_clean_tgt_train.write(file_clean[i]['zh']+"\n")
         elif j['split'] == 'tst':
            raw_clean_src_test.write(file_clean[i]['en'] + "\n")
            raw_clean_tgt_test.write(file_clean[i]['zh'] + "\n")
         elif j['split'] == 'val':
            raw_clean_src_valid.write(file_clean[i]['en'] + "\n")
            raw_clean_tgt_valid.write(file_clean[i]['zh'] + "\n")
     for x,y in enumerate(file_large):
         if y['split'] == 'trn':
             raw_large_src_train.write(file_large[x]['en'] + "\n")
             raw_large_tgt_train.write(file_large[x]['zh'] + "\n")
         elif y['split'] == 'tst':
             raw_large_src_test.write(file_large[x]['en'] + "\n")
             raw_large_tgt_test.write(file_large[x]['zh'] + "\n")
         elif y['split'] == 'val':
             raw_large_src_valid.write(file_large[x]['en'] + "\n")
             raw_large_tgt_valid.write(file_large[x]['zh'] + "\n")