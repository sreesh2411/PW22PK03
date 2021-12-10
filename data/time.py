import matplotlib.pyplot as plt
x = [i+1 for i in range(35)]


with open("withoutK.txt","r") as f:
    t=f.read().split();tl=len(t)
    t1 = list(map(float,t))
    #plt.plot(x,t1)
    print("time withoutK :",sum(t1)/tl)
with open("withK.txt","r") as f:
    t=f.read().split();tl=len(t)
    t2 = list(map(float,t))
    #plt.plot(x,t2)
    print("time withK :",sum(t2)/tl)

"""plt.xlabel('Clients')
plt.ylabel('Time')
plt.title('K means clustering')
plt.show()
"""
