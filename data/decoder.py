import numpy as np
o = 0
def decode_and_aggregate(x,e,c,y):
    global o
    for i in range(8):
        o = 0
        #print(o)
        huffmann_decode(x[i],e[i],c[i])
        #print(o)
    return x+y

def disp(t):
    for i in t:
        print(i.shape)
        print("-")

o = 0
def huffmann_decode(x,e,c):
    global o
    for i in range(len(x)):
        if type(x[i])==np.ndarray or type(x[i])==list:
            huffmann_decode(x[i],e,c)
        else:
            x[i]=c[int(e[o])]
            o+=1
