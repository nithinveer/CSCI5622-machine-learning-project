import os
import json

fet_set =set()


count = 0
for filename in os.listdir('TA-Data-Actual'):
    with open('TA-Data-Actual/{}'.format(filename)) as f:
        data = json.load(f)
        count += len(data)
        for each_data in data:
            for each_sub in  each_data['sub_review']:

                if each_sub not in fet_set:
                    fet_set.add(each_sub)
print(fet_set,count)
