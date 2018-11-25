

# raw = '机构名—医院[M1] 疾病[M2] 科室[M3] 药品[M4] 症状[M5]'
# print(','.join(map(lambda x: "'{}':'{}'".format(x[-3:-1],x[0:-4]),raw.split())))

a = ['Dining','Education','Entertainment','Game','House','Journey','Medical','Physical']

m = '''api.add_resource({}, '/{}')'''

for i in a:
    print(m.format(i,i))