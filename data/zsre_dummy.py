import json

with open('./portability/One_Hop/zsre_mend_eval_portability_gpt4.json') as f:
    data = json.load(f)

for item in data:
    item['cond'] = item['cond'].replace(' >> ' + item['alt'], ' >> ' + 'dummy')
    item['alt'] = 'dummy'
    item['answers'] = ['dummy']
    item['portability']['New Answer'] = 'dummy'

with open('./dummy/dummy_zsre_mend_eval_portability_gpt4.json', 'w') as f:
    json.dump(data, f, indent=4)
