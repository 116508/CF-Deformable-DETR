import json
total=0
mx=0
mp={}
img_total=0
with open(r"E:\experiment\RGBT_detection\DDetr-Single\data\kaist_clean_coco\annotations\visible_test_labels.json",'r',encoding = 'utf-8') as fp:
    res=json.load(fp)
    for i in res["annotations"]:
        id=i["image_id"]
        if id in mp:
            mp[id]=mp[id]+1
        else:
            mp.update({id:1})
    for k,v in mp.items():
        total+=v
        img_total+=1
        mx=max(mx,v)
    print(total,img_total,total/img_total,mx)