import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import csv
import json
import nltk

base_url = 'http://comp20008-jh.eng.unimelb.edu.au:9889/main/'
seed_item = 'Hodg001.html'

seed_url = base_url + seed_item
page = requests.get(seed_url)
soup = BeautifulSoup(page.text, 'html.parser')

visited = []
links = soup.findAll('a')
headline = soup.findAll('h1')

to_visit = []
hrefs = soup.find('p', class_ = 'nextLink')
next_link = hrefs.a['href']
to_visit.append(next_link)

# Task 1 Setup
FILENAME1 = 'task1.csv'
file1 = open(FILENAME1 ,'w+')
writer1 = csv.writer(file1)
writer1.writerow(['url', 'headline'])
# Task 1 Setup

# Task 2 Setup
FILENAME2 = 'task2.csv'
file2 = open(FILENAME2 ,'w+')
writer2 = csv.writer(file2)
writer2.writerow(['url', 'headline', 'team', 'score'])
team_names = []
dataframe = []
with open('rugby.json') as json_file: 
    data = json.load(json_file)
for teams in data['teams']:
    team_names.append(teams['name'])
# Task 2 Setup

# Task 3 Setup
FILENAME3 = 'task3.csv'
file3 = open(FILENAME3 ,'w+')
writer3 = csv.writer(file3)
writer3.writerow(['team', 'avg_game_difference'])
# Task 3 Setup


while(to_visit):
    # Website management
    html = to_visit.pop(0)
    url = base_url + html
    page = requests.get(base_url + html)
    soup = BeautifulSoup(page.text, 'html.parser')
    if html in visited:
        break
    else:
        visited.append(html)
    # Website management
    
    # Data Collection Task 1
    div = soup.find('div', id = 'headline')
    headline = div.h1.text
    writer1.writerow([url, headline])
    # Data Collection Task 1
    
    # Data Collection Task 2
    title = soup.find('h1', attrs = {'class': 'headline'})
    paragraphs = soup.findAll('p', attrs = {'class': None})
    article = ""
    for i in paragraphs:
        article = article + i.text
    raw_headline = r'{0}'.format(title.text)
    raw_article = r'{0}'.format(article)
    tokenize_h = nltk.word_tokenize(raw_headline)
    tokenize_a = nltk.word_tokenize(raw_article)
    headlist = [word for word in tokenize_h if word.isalnum()]
    articlelist = [word for word in tokenize_a if word.isalnum()]
    wordlist = headlist + articlelist
    main_team = ""
    found = 0
    for j in range(len(wordlist)):
        for team in team_names:
            if (len(team.split()) == 1):
                if (wordlist[j] == team):
                    main_team = team
                    found = 1
                    break
            elif (len(team.split()) == 2):
                for i in range(len(team.split())):
                    try:
                        if(wordlist[j] + " " + wordlist[j + 1] == team):
                            main_team = team
                            found = 1
                            break
                    except:
                        break
            elif (len(team.split()) == 3):
                for i in range(len(team.split())):
                    try:
                        if(wordlist[j] + " " + wordlist[j + 1] + " " + wordlist[j + 2] == team):
                            main_team = team
                            found = 1
                            break
                    except:
                        break
        if found == 1:
            break
    scores = []
    pattern = r'\s[0-9][0-9]?[0-9]?-[0-9][0-9]?[0-9]?\s'
    score_h = re.findall(pattern, raw_headline)
    score_a = re.findall(pattern, raw_article)
    scores = score_h + score_a
    if len(scores) == 1 and main_team:
        score = scores[0][1:-1]
        dataframe.append([main_team, score])
        writer2.writerow([url, headline, main_team, score])
    elif len(scores) > 1 and main_team:
        pattern = r'[0-9][0-9]?[0-9]?'
        sum = 0
        sums = []
        for numbers in scores:
            score1 = re.findall(pattern, numbers)[0]
            score2 = re.findall(pattern, numbers)[1]
            sum = int(score1) + int(score2)
            sums.append(sum)
        
        largest = sums.index(max(sums))
        score = scores[largest][1:-1]
        dataframe.append([main_team, score])
        writer2.writerow([url, headline, main_team, score])
    # Data Collection Task 2
    
    # Getting next URL
    hrefs = soup.find('p', class_ = 'nextLink')
    next_link = hrefs.a['href']
    to_visit.append(next_link)
    # Getting next URL

    
# Data Collection Task 3
team = []
score = []
for line in dataframe:
    team.append(line[0])
    pattern = r'[0-9][0-9]?[0-9]?'
    try:
        score1 = re.findall(pattern, line[1])[0]
        score2 = re.findall(pattern, line[1])[1]
        difference = abs(int(score1) - int(score2))
        score.append(difference)
    except:
        continue

df = pd.DataFrame({'team': team, 'score': score})
teams = df['team'].unique()
means = df.groupby(['team']).mean()
final_df = pd.DataFrame({'team': sorted(teams), 'avg_team_difference': means['score'].round(2)})
final_df.to_csv('task3.csv', index = False)
# Data Collection Task 3

# Data Collection Task 4
header = ['Teams']
df = pd.DataFrame(team, columns = header)
df['articles'] = 1
count = df.groupby(['Teams']).count()
count['Teams'] = sorted(teams)
graph = count.sort_values(by = ['articles'], ascending = False).head()
graph.plot.bar(x = 'Teams', y = 'articles', rot = 70, title = 'Number of articles written about each team')
plt.savefig('task4.png', bbox_inches = 'tight')
plt.show()
# Data Collection Task 4

# Data Collection Task 5
FNAME = 'task3.csv'
avg = []
for line in csv.reader(open(FNAME)):
    if (line[1].replace('.', '', 1).isdigit()):
        n = float(line[1])
        avg.append(n)
data = count
data['average'] = avg
plt.scatter(x = data['articles'], y = data['average'])
art = np.array(data['articles'])
ave = np.array((data['average']))
m, b = np.polyfit(art, ave, 1)
plt.plot(art, m * art + b)
plt.title("Scatter plot of articles written vs average game difference")
plt.xlabel("Articles Written")
plt.ylabel("Average Game Difference")
plt.savefig('task5.png')
plt.show()
# Data Collection Task 5