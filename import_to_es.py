import os
import json
import  requests
post_request_headers = {
"Content-Type": "application/json"
}





for filename in os.listdir('TA-Data-Actual'):
    with open('TA-Data-Actual/{}'.format(filename)) as f:
        data = json.load(f)
        for each_data in data:
            _id = filename.replace('.json','') + "-"+each_data['page_url'].split('-r')[1]
            each_data['hotel_id'] =filename.replace('.json','')
            each_data['review_id'] = each_data['page_url'].split('-r')[1]
            insert_request = requests.post(
                url="http://13.126.22.216:3718/ta_info/_doc/{}".format(_id),
                data=json.dumps(each_data),
                headers=post_request_headers)
            print(insert_request.json())