import tensorflow as tf
import numpy as np
import zipfile

def prune(before,after,p):
    global c
    f=after-before
    for i in [0,2,4,6]:
        # ct=np.count_nonzero(f[i])
        # print(ct)
        updates=f[i]
        all_=abs(updates).flatten()
        all_=all_[all_!=0]
        l=int(len(all_)*p)
        k=max(np.partition(all_,l)[:l])
        updates[abs(updates)<=k]=int(0)
        f[i]=updates
    #     print(ct-np.count_nonzero(f[i]))
    #     print("---")
    # print("###")
    return f+before


def disp(t):
    for i in t:
        print(i.shape)
        print("-")



def kmeans2(f):
    for i in [0,2,4,6]:
        x=f[i].flatten()
        num_points = len(x)
        dimensions = 1
        points = x.reshape(-1,1)

        def input_fn():
          return tf.compat.v1.train.limit_epochs(
              tf.convert_to_tensor(points, dtype=tf.float32), num_epochs=1)

        num_clusters = 3
        kmeans = tf.compat.v1.estimator.experimental.KMeans(
            num_clusters=num_clusters, use_mini_batch=False)

        # train
        num_iterations = 6
        previous_centers = None
        for _ in range(num_iterations):
              kmeans.train(input_fn)
              #cluster_centers = kmeans.cluster_centers()
              #if previous_centers is not None:
            #      print('delta:', cluster_centers - previous_centers)
              #previous_centers = cluster_centers
              #print('score:', kmeans.score(input_fn))
        print('cluster centers:', kmeans.cluster_centers())

        #map the input points to their clusters
        #cluster_indices = list(kmeans.predict_cluster_index(input_fn))
        #print(cluster_indices,cluster_centers)
        """for i, point in enumerate(points):
          cluster_index = cluster_indices[i]
          center = cluster_centers[cluster_index]
          print('point:', point, 'is in cluster', cluster_index, 'centered at', center)"""


def kmeans(f):
    out = []
    cc = []
    with tf.Session() as sess:
        for i in range(8):
            x=f[i].flatten()
            x = x[x!=0] #
            y=x.reshape(-1,1)
            clusters_n = 3
            iteration_n = 10
            points = tf.constant(y)
            centroids = tf.Variable(tf.slice(tf.random_shuffle(points), [0, 0], [clusters_n, -1]))
            points_expanded = tf.expand_dims(points, 0)
            centroids_expanded = tf.expand_dims(centroids, 1)
            distances = tf.reduce_sum(tf.square(tf.subtract(points_expanded, centroids_expanded)), 2)
            assignments = tf.argmin(distances, 0)
            means = []
            for c in range(clusters_n):
                means.append(tf.reduce_mean(
                  tf.gather(points,
                            tf.reshape(
                              tf.where(
                                tf.equal(assignments, c)
                              ),[1,-1])
                           ),reduction_indices=[1]))
            new_centroids = tf.concat(means, 0)
            update_centroids = tf.assign(centroids, new_centroids)
            init = tf.global_variables_initializer()

            sess.run(init)
            for step in range(iteration_n):
                [_, centroid_values, points_values, assignment_values] = sess.run([update_centroids, centroids, points, assignments])

            c = centroid_values[~np.isnan(centroid_values)]
            #print(len(c),c)
            final_c=np.append(c,np.array([0,]))
            c1=final_c
            cc.append(c1)
            out.append(call_huffmann_bro(f[i],final_c))

    """with open("updates.txt","w") as p:
        for i in out:
            p.write("".join(map(str,i)))
            p.write("\n")
    with open("updatesc.txt","w") as p:
        for i in cc:
            p.write("".join(map(str,i)))
            p.write("\n")"""

    #out = ["".join(map(str,i)) for i in out]
    #print(out)
    #np.save("test.txt",out)
    tf.keras.backend.clear_session()

    return out,cc
    "return f"

def call_kmeans(before,after):
        f=after-before
        # Layer 6
        f,cc=kmeans(f)
        # print(np.count_nonzero(f[6]))

        #print(f[0][0][0])
        #print((before - before + f)[0][0][0])
        #t_=before - before + f
        #print("Centroids per 2 layers")
        #for j in range(8):
        #    print(np.unique(f[j]))
        c1 = 3 # +1 (0) = 4
        return c1,cc,f
        #return f

def update_updates_with_centeroids(x,c):
    for i in range(len(x)):
        if type(x[i])==np.ndarray or type(x[i])==list:
            update_updates_with_centeroids(x[i],c)
        else:
            x[i]=find_nearest_centroid(c,x[i])



def find_nearest_centroid(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def convert(t1 , t2):
    t1=np.array(t1)
    t2=np.array(t2)
    disp(t1)
    disp(t2)




import json

def call_huffmann_bro(k,c):
    huffmann_encode(k,c)

    t="".join(map(str,map(int,k.flatten())))
    return t
    #huffmann_decode(k,c)

def replace_with_key(x,c):
        array = np.asarray(c)
        return int((np.abs(c - x)).argmin())

def huffmann_encode(x,c):
    for i in range(len(x)):
        if type(x[i])==np.ndarray or type(x[i])==list:
            huffmann_encode(x[i],c)
        else:
            x[i]=replace_with_key(c,x[i])

o = 0
def huffmann_decode(x,e,c):
    global o
    for i in range(len(x)):
        if type(x[i])==np.ndarray or type(x[i])==list:
            huffmann_decode(x[i],e,c)
        else:
            x[i]+=c[int(e[o])]
            o+=1


if __name__=="__main__":
    first = np.load("first.npy")
    before = np.load("before.npy")
    after = np.load("after.npy")
    disp(first)
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
    #kmeans2(f)
    #print(t3[0][0][0])
    #print((new_first-first)[0][0][0])
    #print(t3[0][0][0]==(new_first-first)[0][0][0])
    #print(np.array_equal(t3[0][0][0], (new_first-first)[0][0][0]))
    #print(sorted(c1)==sorted(np.unique(t3[6].flatten())))
    #np.save("sparsification",t3)

# print(sys.getsizeof(before)*3136)
# print(sys.getsizeof(after)*3136)
