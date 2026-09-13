strategies={
 'correct_general': {'public':1.0,'hidden':1.0,'speed':0.6},
 'hardcode_public': {'public':1.0,'hidden':0.1,'speed':1.0},
 'fast_wrong':      {'public':0.2,'hidden':0.2,'speed':1.0},
}
for name,s in strategies.items():
    proxy=0.8*s['public']+0.2*s['speed']
    true=0.8*s['hidden']+0.2*s['speed']
    print(name,'proxy=',round(proxy,2),'true=',round(true,2))
