import requests
import json


url = "https://www.google.com/inputtools/request?ime=handwriting"

a = [[ 0.15709319,  7.31667334],
       [ 0.25194673,  7.41773017],
       [ 0.33590762,  7.02145544],
       [-0.17153779,  5.43455189],
       [-1.75689605, -1.41714019],
       [-1.20899564,  1.21314914],
       [-0.75209838,  2.73236963],
       [-0.10226573,  2.67594858],
       [ 1.62571135,  1.51761857],
       [ 2.17077607,  0.36856216],
       [ 1.96798926, -0.19000122],
       [ 1.27206163, -1.40705569],
       [ 0.4946375 , -2.2518482 ],
       [-0.74460714, -2.46008298],
       [-1.45797739, -2.12824824],
       [-1.75659708, -1.71644507],
       [-1.70720647, -0.42501755],
       [-1.72959306,  0.31873726]]

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