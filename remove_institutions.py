import re
import json

CONFERENCE = 'cvpr'
YEAR = '2012'
uri = "paper_names/"+CONFERENCE+YEAR+"_papers.json"

data = json.load(open(uri))
for i in range(len(data)):
	authors = data[i]['Authors']
	institutions = re.findall("\(.*?\)", authors)
	for inst in institutions:
		data[i]['Authors'] = data[i]['Authors'].replace(inst,"")
	data[i]['Authors'] = ", ".join([x.strip(" ") for x in data[i]['Authors'].split(",")])

json.dump(data, open(uri+"_new", 'w'), indent=4)