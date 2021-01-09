import math
import numpy
from copy import deepcopy
from itertools import combinations
import cv2
from PIL import Image
from skimage import io
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import json
import ffmpeg
import pickle
        
#wrapper for vectors that makes them hashable
class vec:
    def __init__(self,l):
        self.v = l
        self.n = 0
    def __hash__(self):
        return int(sum(self.v))
    def __eq__(self,other):
        if isinstance(other,vec):
            return self.v == other.v
        else:
            return False
    def __getitem__(self,num):
        return self.v[num]
    def __setitem__(self,num,dat):
        self.v[num] = dat
    def __iter__(self):
        self.n = 0
        return self
    def __str__(self):
        return 'vec'
    def __repr__(self):
        return 'vec'
    def __next__(self):
        if self.n>=len(self.v):
            raise StopIteration
            return
        ret = self.v[self.n]
        self.n+=1
        return ret
    def __len__(self):
        return len(self.v)
    def __reduce__(self):
        return (vec,(self.v,))

def dist(v1,v2):
    return cosine_similarity([v1.v],[v2.v])[0][0]
def scale(scalar,vecc):
    ret = []
    for i in vecc:
        ret.append(i*scalar)
    return vec(ret)
def add(v1,v2):
    ret = []
    for i,j in zip(v1,v2):
        ret.append(i+j)
    return vec(ret)
def addAll(vecs):
    ret = []
    trans = numpy.transpose(vecs)
    for row in trans:
        ret.append(sum(trans))
    return vec(ret)
def average(v1,v2):
    return add(scale(.5,v1),scale(.5,v2))
def nCr(n,r):
    f = math.factorial
    return f(n) // f(r) // f(n-r)
def generate_representative_set(vecs,size,shape):
    ret = deepcopy(vecs)
    while len(ret)>size:
        #find closest two vectors
        closest = ()
        closest_dist = -1
        count = 0
        p = nCr(len(ret),2)
        dists = {}
        for i,v in combinations(ret,2):
            count+=1
            d = None
            try:
                d = dists[(i,v)]
                print(str(p-count)+' , '+str(len(ret)-size)+' , REUSE')
            except:
                d = dist(i,v)
                dists[(i,v)] = d
                print(str(p-count)+' , '+str(len(ret)-size))
            if(d>closest_dist):
                closest = (i,v)
                closest_dist = d
        #replace those two vectors with their average
        avg = average(closest[0],closest[1])
        ret.remove(closest[0])
        ret.remove(closest[1])
        ret.append(avg)
    return ret
def recursive_build(model,repset,time_len,depth):
    if depth==time_len+1:
        return
    for k in repset:
        submodel = {}
        model[k] = [0,submodel]
        recursive_build(submodel,repset,time_len,depth+1)
    return
def build_model(data,granularity,time_len,shape):
    repset_l = generate_representative_set(data,granularity,shape)
    print('built repset')
    repset = [vec(x) for x in repset_l]
    model = (time_len,{})
    recursive_build(model[1],repset,time_len,0)
    return model
def recursive_normalize(mod):
    sm = 0
    for k in mod:
        sm+=mod[k][0]
    for k in mod:
        mod[k][0]/=sm
    for k in mod:
        recursive_normalize(mod[k][1])
def train_model(data,model):
    time_len = model[0]
    mod = model[1]
    p = len(data)-time_len
    for i in range(len(data)-time_len):
        print(p-i)
        recursive_train(mod,data,0,time_len,1,i)
    #normalize the probabilities in the model
    recursive_normalize(mod)
    return mod
def recursive_train(model,data,depth,time_len,modifier,window):
    if depth==time_len+1:
        return
    img = data[depth+window]
    for k in model:
        d = dist(k,img)
        model[k][0]+=d*modifier
        recursive_train(model[k][1],data,depth+1,time_len,modifier*d,window)
def predict(data,model,time_len):
    dat = data[-time_len:]
    pred_map = predict_recursive(model,dat,0,1)
    # normalize prediction
    sm = 0
    for key in pred_map:
        sm+=pred_map[key]
    for key in pred_map:
        pred_map[key]/=sm
    print(pred_map)
    ret = None
    for key in pred_map:
        if ret == None:
            ret=scale(pred_map[key],key)
        else:
            ret = add(ret,scale(pred_map[key],key))
    return ret

def combine_maps(m1,m2):
    for key in m2:
        if key in m1:
            m1[key]+=m2[key]
        else:
            m1[key] = m2[key]
    return m1
def predict_recursive(model,data,depth,modifier):
    if depth==len(data):
        ret = {}
        for k in model:
            ret[k] = modifier*model[k][0]
        return ret
    else:
        ret = {}
        for k in model:
            newmod = (modifier*model[k][0])/4+dist(k,data[depth])
            ret = combine_maps(ret,predict_recursive(model[k][1],data,depth+1,newmod))
        return ret
def predict_multiple(model,inp,time_len,frames):
    ret_v = inp
    for i in range(frames):
        print(i)
        ret_v.append(predict(ret_v,model,time_len))
    ret = []
    for v in ret_v:
        ret.append(v.v)
    return ret


vidcap = cv2.VideoCapture('p.3gp')
success,image = vidcap.read()
images = []
while success: 
  success,image = vidcap.read()
  if(success):
      images.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
shape = images[0].shape
data = []
for d in images:
    data.append(vec(d.ravel().tolist()))

model = build_model(data,100,4,shape)
print('built model')
model = train_model(data,model)
print('trained model')
pickle.dump(model,open('out.p','wb'))
#model = pickle.load(open('out.p','rb'))
pred = predict_multiple(model,data[35:45],5,30)
count = 0
for pi in pred:
    predImage = numpy.array(pi).reshape(shape)
    cv2.imwrite('imgs/'+str(count)+'.jpg',predImage)
    count+=1
