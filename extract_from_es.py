import os
import json
import  requests
post_request_headers = {
"Content-Type": "application/json"
}

months = ["January","February","March","April","May","June","July","August","September","October","November"]
def getData(hotelId='1418811'):
    response = {}
    for each_month in months:
        es_query = '{"query":{"bool":{"must":[{"match":{"hotel_id":"'+hotelId+'"}},{"range":{"date":{"lte":"'+each_month+" 2019"+'","gte":"'+each_month+" 2019"+'","format":"MMM yyyy||yyyy"}}}]}},"size": 2000,"aggs":{"avg_rate":{"avg":{"field":"rating"}}}}'
        # print(es_query)
        es_response = requests.post(
            url="http://13.126.22.216:3718/ta_infov2/_search",
                data=es_query,
                headers=post_request_headers).json()
        reviews_response= {}
        reviews=[]
        if 'hits' in es_response and es_response['hits']['total'] >0:
            for each_hit in es_response['hits']['hits']:
                reviews.append(each_hit['_source']['review'])

        reviews_response['reviews'] = reviews
        reviews_response['avg_rating'] = 0
        if len(reviews)> 0:
            reviews_response['avg_rating']= es_response['aggregations']['avg_rate']['value']
        response[each_month] = reviews_response

    return response



