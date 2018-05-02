# Boston Bombing tweets clustering

import pandas as pd
import numpy as np
import nltk
import sys
import math

def read_tweets(a):  
    path=a
    df=pd.read_json("Tweets.json",lines=True)
    df=df[['id','text']]
    df['text']=df['text'].str.replace("http\S+|www.\S*","",case=False)
    df['text']=df['text'].str.replace("([@])\w+","")
    df['text']=df['text'].str.replace("RT","")
    df['text']=df['text'].str.replace(":","")
    df['text']=df['text'].str.replace("-","")
    df['text']=df['text'].str.replace("([0-9])\w+","")
    df['text']=df['text'].str.lower()
    df['text']=df['text'].str.replace("#","")
    df['text']=df['text'].str.replace("|","")
    df['text']=df['text'].str.replace("\"","")
    df['text']=df['text'].str.replace(".","")
    df['text']=df['text'].str.replace("(","")
    df['text']=df['text'].str.replace(")","")
    return df


def jacquard_dist(tweet1,tweet2,df):
    x=df['text'].loc[df['id'] == int(tweet1)].iloc[-1]
    y=df['text'].loc[df['id'] == int(tweet2)].iloc[-1]
    n=0
    w1=set(x.lower().split())-set(nltk.corpus.stopwords.words('english'))
    w2=set(y.lower().split())-set(nltk.corpus.stopwords.words('english'))
    for word in w2:
        if word in w1:
            n += 1
    Jacquard_dist=1-(len(w1.intersection(w2))/len(w1.union(w2)))
    return Jacquard_dist
    

def get_InitialSeeds(x):
    path=x
    c= open(path, 'r')
    c=[line.strip(',\n') for line in c.readlines()]
    return c

	
def new_centroids(c,clusters,df):
    curmin=999
    for i in range(len(c)):
        for j in clusters[i]:
            min=jacquard_dist(j,c[i],df)
            if min<curmin:
                curmin=min
                index=j
                c[i]=index
    return c



def calc(a,x,df):
    curmin=9999
    for i in range(len(x)):
        dist=jacquard_dist(a,x[i],df)
        if dist<curmin:
            curmin=dist
            index=i
    return index,a


def sum_squared_errors(c,clusters,df):
    sum=0
    for i in range(len(c)):
        for j in clusters[i]:
            dist=jacquard_dist(j,c[i],df)
            sum+=pow((dist),2)           
    return sum
       

def main(argv):
    k=int(sys.argv[1])
    centroids=[]
    centroids=get_InitialSeeds(sys.argv[2])
    df=read_tweets(sys.argv[3])
    output_path=sys.argv[4]
    while True:
        clusters={}
        for i in range(len(centroids)):
            clusters[i]=[]
        for i in range(len(df)):
            clus_indx,twt_id=calc(df.iloc[i,0],centroids,df)
            clusters[clus_indx].append(twt_id)
        old_centroids = list(centroids)
        centroids=new_centroids(centroids,clusters,df)
        if centroids==old_centroids:
            break
    sse=sum_squared_errors(centroids,clusters,df) 
    with open(output_path, "a") as file:
        for i in clusters:
            file.write(str(i+1)+ "    ")
            file.writelines(["%s,  " % item  for item in clusters[i]])
            file.write("\n")
            file.write("\n")
            file.write("\n")
        file.write("SSE=  "+str(sse))
         


if __name__ == "__main__":
    main(sys)    
    
    




        

