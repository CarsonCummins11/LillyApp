import re
import random
import json
import time,sys
class autocomplete:
    def __init__(self,n,model={}):
        self.ngrams = n
        self.model = model
        self.vectorization = {}
    @staticmethod
    def load(mod_file):
        txt = ''
        for line in open(mod_file):
            txt+=line
        k = txt.split('#')
        ngrams = int(k[0])
        model = json.loads(k[1])
        return autocomplete(ngrams,model=model)
    def train(self,txt_file,out_file):
        txt = ''
        for line in open(txt_file,'r'):
            txt+=line
        txt = txt.split()
        self.model = {}
        for i in range(len(txt)):
        #normalize the words, remove punctuation, capitals, etc
            txt[i] = self.normalize(txt[i])
        #construct tree representing training text with given probabilities
        for i in range(len(txt)-self.ngrams):
            subtree = self.model
            for j in range(self.ngrams):
                if txt[i+j] in subtree:
                   subtree[txt[i+j]][0]+=1
                else:
                    subtree[txt[i+j]] = [1,{}]
                subtree = subtree[txt[i+j]][1]
        f = open(out_file,'w+')
        f.write(str(self.ngrams)+'#'+json.dumps(self.model))
        f.close()
    def predict(self,txt,opt=''):
        t = txt.split()[max(0,len(txt.split())-self.ngrams) :]
        for i in range(len(t)):
            t[i] = self.normalize(t[i])
        best = {}
        stree = {}
        for i in reversed(range(len(t)+1)):
            sub=t[len(t) - i :]
            subtree = self.model
            succ = True
            for k in range(len(sub)):
                if sub[k] in subtree:
                    ree = subtree[sub[k]][1]
                    if ree is not None and len(ree) != 0:
                        subtree=ree
                else:
                    succ=False
                    break
            if(succ):
                for key in subtree:
                    if key in stree:
                        stree[key]+=(i**(self.ngrams*1.2))*subtree[key][0]
                    else:
                        stree[key]=(i**(self.ngrams*1.2))*subtree[key][0]
        choices = []
        for key in stree:
            for m in range(stree[key]):
                choices.append(key)
        choices = [x for x in choices if x != opt]
        if len(choices) == 0:
            return ''
        return random.choice(choices)
    def normalize(self,word):
        word = word.lower()
        return re.sub(r'[^a-zA-Z]', '',word)
    def predict_multiple(self,start,add):
        ret = start
        for j in range(add):
            final = ret.split()[-1]
            m = self.predict(ret,opt=final)
            ret+=' '+m
            if m=='':
                break
        return ret
    def reason(self):
        starters = ['Lilly, I love you because', 'Youre the cutest because', 'youre the best because','youre my favorite because','youre what i think about all the time because','you put a smile on my face because','youre the best girl ive ever met because','youre number one because','I like that you','I like when you','I love when you','I love that you']
        return self.predict_multiple(random.choice(starters),15)
ac = autocomplete(5)
ac.train('big.txt','out.txt')
print(ac.predict_multiple('To Sherlock Holmes she is always the woman',20))