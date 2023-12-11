from http import server
from http.server import HTTPServer, BaseHTTPRequestHandler
import pickle
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from googleapiclient.discovery import build
import json
import cgi

api_key='AIzaSyA8F71DrUn17b1IKVhNtOX98GHeTshWwzo'

def get_comments(youtube,id):
    comment_list=[]
    request = youtube.commentThreads().list(
        part="snippet",
        maxResults=50,
        moderationStatus="published",
        textFormat="plainText",
        videoId=id
    )
    response = request.execute()
    for item in response['items']:
        comment_list.append(item['snippet']['topLevelComment']['snippet']['textDisplay'])
    return comment_list

def clean(comments):
    all_reviews= list()
    lines = comments
    for text in lines:
        #1
        text = text.lower()
        #2
        pattern = re.compile('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        text = pattern.sub('', text)
        emoji = re.compile("["
            u"\U0001F600-\U0001FFFF"  # emoticons
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs
            u"\U0001F680-\U0001F6FF"  # transport & map symbols
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        text = emoji.sub(r'', text)
        #3
        text = re.sub(r"[,.\"!@#$%^&*(){}?/;`~:<>+=-]", "", text)
        #4
        tokens = word_tokenize(text)
        #5
        words = [word for word in tokens if word.isalpha()]
        #6
        stop_words = set(stopwords.words("english"))
        stop_words.discard("not")
        words = [w for w in words if not w in stop_words]
        #7
        SB = SnowballStemmer(language='english')
        words = [SB.stem(w) for w in words if not w in stop_words]
        #8
        words = ' '.join(words)
        if(words!=''):
            all_reviews.append(words)
    return all_reviews

def processComments(comments):
    good=0
    bad=0
    english_comments=[]
    clean_comments=clean(comments)
    for x in clean_comments:
        lang = detect(x)
        if(lang=='en'):
            english_comments.append(x)
    TVfilename = './Server-Side/finalized_TFIDF.sav'
    TV = pickle.load(open(TVfilename, 'rb'))
    X = TV.transform(english_comments).toarray()
    filename = './Server-Side/finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    y= loaded_model.predict(X)
    for x in y:
        if x==1: good+=1
        if x==0: bad+=1
    return 100*(good)/(good+bad)

def get_videos(youtube,search):
    video_list=[]
    request = youtube.search().list(
        part="snippet",
        maxResults=5,
        q=search,
        type="video"
    )
    response = request.execute()
    for item in response['items']:
        vid ={}
        vid['id']=item['id']['videoId']
        vid['title']=item['snippet']['title']
        vid['sentiment']=processComments(get_comments(youtube,vid['id']))
        request = youtube.videos().list(
            part="snippet,statistics",
            id=vid['id'],
            maxResults=1
        )
        response2=request.execute()
        for item in response2['items']:
            # vid['desc']=item['snippet']['description']
            vid['img']=item['snippet']['thumbnails']['default']['url']
            vid['views']=item['statistics']['viewCount']
            vid['likes']=(100*int(item['statistics']['likeCount']))/(int(item['statistics']['dislikeCount'])+int(item['statistics']['likeCount']))
        video_list.append(vid)
    return video_list

def generate_response(search_query,sort_query):
    youtube = build('youtube','v3',developerKey=api_key)
    vids=get_videos(youtube,search_query)
    if(sort_query=='Views'):
        vids.sort(key=lambda x: x['views'])
        vids.reverse()
        for vid in vids:
            print(vid['title'])
            print(vid['views'])
    if(sort_query=='Sentiment'):
        vids.sort(key=lambda x: x['sentiment'])
        vids.reverse()
        for vid in vids:
            print(vid['title'])
            print(vid['sentiment'])
    if(sort_query=='Likes'):
        vids.sort(key=lambda x: x['likes'])
        vids.reverse()
        for vid in vids:
            print(vid['title'])
            print(vid['likes'])
    return vids


class requestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        fields=self.rfile.read(int(self.headers['content-length']))
        data=json.loads(fields.decode('utf-8'))
        print(data['Search'])
        print(data['SortBy'])
        result_list=generate_response(data['Search'],data['SortBy'])
        self.send_response(200)
        self.send_header('content-type','application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(result_list).encode())

def main():
    PORT=1234;
    server=HTTPServer(('',PORT),requestHandler)
    print('server running on port',PORT)
    server.serve_forever()

if __name__ == '__main__':
    main()