import json
from bs4 import BeautifulSoup
import requests
from math import floor
import urllib3


class_to_rating = {
    "bubble_10": 1,
    "bubble_15": 1.5,
    "bubble_20": 2,
    "bubble_25": 2.5,
    "bubble_30": 3,
    "bubble_35": 3.5,
    "bubble_40": 4,
    "bubble_45": 4.5,
    "bubble_50": 5
}


ta_url = 'http://api.tripadvisor.com/api/partner/2.0/location/{}?key=61F1644563794354A518190F4505BA38'

with open('GiataCode-MHId-TAId.json') as f:
  data = json.load(f)
for each_data in data:
    ta_id = each_data['tripAdvisorId']
    ta_raw_info = requests.get(ta_url.format(ta_id)).json()
    if 'reviews' in ta_raw_info:
        review_url = (ta_raw_info['reviews'][0]['url'].split('-r')[0])
        print(review_url)
    review_count = 0
    if 'review_rating_count' in ta_raw_info:
        if '1'  in ta_raw_info['review_rating_count']:
            review_count += int(ta_raw_info['review_rating_count']['1'])
        if '2'  in ta_raw_info['review_rating_count']:
            review_count += int(ta_raw_info['review_rating_count']['2'])
        if '3'  in ta_raw_info['review_rating_count']:
            review_count += int(ta_raw_info['review_rating_count']['3'])
        if '4'  in ta_raw_info['review_rating_count']:
            review_count += int(ta_raw_info['review_rating_count']['4'])
        if '5'  in ta_raw_info['review_rating_count']:
            review_count += int(ta_raw_info['review_rating_count']['5'])
        review_count = (floor(review_count/5))
    reviews_obj = []
    # review_count = 5
    for i in range(1,review_count+1):
        review_page_url = review_url.replace('ShowUserReviews','Hotel_Review') + '-Reviews-or' + str((i-1)*5)
        print(review_page_url)
        try :
            http = urllib3.PoolManager()
            res = http.request("GET", review_page_url)
            soup = BeautifulSoup(res.data, features="html.parser")
            x = soup.find_all("div", {"class": "cPQsENeY"})
            user_reviews = []
            for v_0 in x[1:]:
                # print(v_0.text)
                user_reviews.append(v_0.text)
            user_review_dates = []
            x1 = soup.find_all("span", {"class": "hotels-review-list-parts-EventDate__event_date--CRXs4"})
            for v_1 in x1:
                # print(v_1.text)
                user_review_dates.append(v_1.text)
            user_review_titles = []
            x2 = soup.find_all("a", {"class": "hotels-review-list-parts-ReviewTitle__reviewTitleText--3QrTy"})
            for v_2 in x2:
                # print(v_2.text)
                user_review_titles.append(v_2.text)
            user_review_ids = []
            x3 = soup.find_all("div", {"class": "hotels-review-list-parts-SingleReview__mainCol--2XgHm"})
            for v_3 in x3:
                # print(v_3['data-reviewid'])
                user_review_ids.append(v_3['data-reviewid'])
            index = 0
            for each_review in user_review_ids:
                ind_review = {}
                ind_review['title'] = user_review_titles[index]
                ind_review['review'] = user_reviews[index]
                ind_review['date'] = user_review_dates[index].replace("Date of stay: ","")
                review_page_url = review_url + '-r' +str(each_review)
                # print(review_page_url)
                ind_review['page_url'] =review_page_url
                ind_review['sub_review'] = {}
                ind_review['travel_purpose'] = ''

                try :
                    http1 = urllib3.PoolManager()
                    res2 = http1.request("GET", review_page_url)
                    soup1 = BeautifulSoup(res2.data,features="html.parser")
                    x3 = soup1.find("div", {"class": "rating-list"})
                    e1 = soup1.select(".ui_bubble_rating")
                    # print((e1))
                    e3 = []
                    e3.append(e1[0])
                    key_list = []
                    # for v_4 in x3:
                    #    print(v_4.text)
                    for v_4 in e3:
                        # print(v_4)
                        ind_review['rating'] = (class_to_rating[v_4["class"][1]])
                    # exit(0)

                    travel_purpose = ''
                    try :
                        x33 = x3.find("div", {"class": "recommend-titleInline"})
                        travel_purpose = str(x33).split('>')[-2].replace("</div","")
                        p = 1
                    except Exception as ex:
                        print("Purpose Exception " + str(ex))
                        p = 0
                    ind_review['travel_purpose'] = travel_purpose
                    e22 = x3.select("div", {"class": "ui_bubble_rating"})
                    # p = 1
                    sub_review = {}
                    while p < len(e22):
                        typee = e22[p]
                        rate = e22[p + 1]
                        #     print(typee,rate)
                        # print(class_to_rating[typee['class'][1]],
                        #       str(rate).split('"')[2].split('<')[0].replace('>', ''))
                        sub_review[str(rate).split('"')[2].split('<')[0].replace('>', '')] = int(class_to_rating[typee['class'][1]])
                        p+=2
                    ind_review['sub_review'] = sub_review
                except Exception as ex:
                    print("Sub reviews"+str(ex))
                reviews_obj.append(ind_review)
                index+=1
            # exit(0)
        except Exception as ex:
            print(str(ex))

    print(ta_id)
    with open('TA-Data-Latest/'+str(ta_id)+'.json', 'w') as json_file:
        json.dump(ta_raw_info, json_file)
    with open('TA-Data-Actual/'+str(ta_id)+'.json', 'w') as json_file:
        json.dump(reviews_obj, json_file)
    # print(reviews_obj)
    # exit(0)

print(len(data))