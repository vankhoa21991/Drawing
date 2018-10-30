import requests
import json


url = "https://www.google.com/inputtools/request?ime=handwriting"

a = [[ 1.0244783 ,  4.55478407],
       [ 1.07799002,  4.65040763],
       [ 1.44357909,  4.30527209],
       [ 1.82654514,  3.20171183],
       [-0.3185472 , -3.60697292],
       [ 0.20712973, -0.35878202],
       [ 1.40142692,  1.60749975],
       [ 1.97662814,  1.6044559 ],
       [ 3.23470617,  0.42707449],
       [ 3.92251402, -1.20712084],
       [ 3.89902265, -1.98804528],
       [ 2.95060327, -3.4373501 ],
       [ 1.58761727, -4.21686734],
       [ 0.44562147, -4.37247765],
       [-0.08816165, -4.14365583],
       [-0.31749666, -3.83760425],
       [-0.55663498, -2.5989708 ],
       [-1.25018483, -1.36570236]]

N = 100
M = 500

a_new = [[x*N+M for x in y] for y in a]
x,y = zip(*a_new)
y_new = [i*-1 for i in y]
_input = [[x,y_new]]

data = { "device":"Chrome/19.0.1084.46 Safari/536.5",
         "options":"enable_pre_space",
         "requests":[{"writing_guide":{
     "writing_area_width":1920,
     "writing_area_height":617},
    "ink": _input,
     "language":"en"}]}

headers = {'Content-type': 'application/json'}

r = requests.post(url, data=json.dumps(data), headers=headers)

print(r.status_code)

print(r.json())