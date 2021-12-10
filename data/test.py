
# for i in range(128):
#     for j in range(62):
#         if before[6][i][j]!=after[6][i][j]:
#             print(before[6][i][j],after[6][i][j])
#         else:
#             print("True")
# print(np.count_nonzero(a1[6]),np.count_nonzero(a2[6]))
import timeit
with open("standard.txt","w") as f:
    for i in first:
        t=list(i.flatten())
        f.write(",".join(map(str,t)))
        f.write("\n")
        import json
c1,t3,cc=call_kmeans(before,after)
