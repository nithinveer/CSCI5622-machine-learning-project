import json
import requests
ta_url = 'http://api.tripadvisor.com/api/partner/2.0/location/{}?key=61F1644563794354A518190F4505BA38'

with open('GiataCode-MHId-TAId-Copy.json') as f:
  data = json.load(f)
for each_data in data:
    ta_id = each_data['tripAdvisorId']
    ta_raw_info = requests.get(ta_url.format(ta_id)).json()
    print(ta_id)
    with open('TA-Data/'+str(ta_id)+'.json', 'w') as json_file:
        json.dump(ta_raw_info, json_file)
    # exit(0)

print(len(data))