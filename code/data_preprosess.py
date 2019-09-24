"""
@author: zhixin lai
@contact: laizhixin16@gmail.com
"""
import re as re

# 检查是否有共作 方式一：
# input authors1and authors2 are authors of two papers"list",sim_num is the number of silimar co-author
# output is 0or1 1 means has the same author
def coau_group(authors1, authors2, sim_num):
    k = 0
    kk = 0
    authors1_temp = []
    authors2_temp = []
    for i in authors1:
        authors1_temp.append(i)
    for i in authors2:
        authors2_temp.append(i)
    for i in authors1_temp:
        for j in authors2_temp:
            if i == j:
                k = k+1
                break
    if k > sim_num:    # 该参数可以修改，进而提高效率， 是否有两个共作及以上
        kk = 1
    return kk

# 预处理作者名字方式一：简写前两位
# preprocess authors' name/ input is the authors' name"str" / output is the name after preproce
def name_prepro(text):
    text = text.lower()
    text = re.sub("\.", " ", text)
    text = re.sub("\,", " ", text)
    text = re.sub("\'", " ", text)
    text = re.sub("-", "", text)
    text = re.sub("_", " ", text)
    text = re.sub(' +', ' ', text)
    text = text.strip('._ ')
    text = text.split(" ")
    text_temp = []
    i=len(text)
    if i <= 2:
        text[0] = text[0][0]
    elif i == 3:
        text[0] = text[0][0]
        text[1] = text[1][0]
    else:
        text_temp.append(text[0][0])
        text_temp.append(text[1][0])
        text_temp.append(text[i-1])
        text = text_temp
    text_tran = ' '.join(text)  # list to string
    return text_tran

# 预处理作者名字方式二：简单去除字符，但不修改其他
# preprocess authors' name/ input is the authors' name"str" / output is the name after preproce
def name_prepro(text):
    text = text.lower()
    text = re.sub("\.", " ", text)
    text = re.sub("\,", " ", text)
    text = re.sub("\'", " ", text)
    text = re.sub("-", "", text)
    text = re.sub("_", " ", text)
    text = re.sub(' +', ' ', text)
    text = text.strip('._ ')
    text = text.split(" ")
    text_tran = ' '.join(text)  # list to string
    return text_tran

# title预处理：去掉关干扰符号以及高频词
# preprocess title / input is the authors'title  "str" / output is the title "str" after preproce
def title_prepro(text):

    text = text.lower()
    text = re.sub("\."," ",text)
    text = re.sub("\,"," ",text)
    text = re.sub("\'"," ",text)
    text = re.sub("_"," ",text)
    text = re.sub("-"," ",text)
    text = re.sub(" +", " ", text)
    text = text.strip('._ ')
    text = text.split(" ")
    highfre_org = []
    text_temp = copy.deepcopy(text)
    for i in text_temp:
        for j in highfre_org:
            if i == j:
                text.remove(i)
    text = ' '.join(text)
    return text


#preprocess authors' org/ input is the authors' org"str" / output is the org"str" after preproce
#地址预处理：去掉关干扰符号以及高频词
def org_prepro(text):

    text=text.lower()
    text= re.sub("\."," ",text)
    text= re.sub("\,"," ",text)
    text= re.sub("\'"," ",text)
    text= re.sub("_"," ",text)
    text= re.sub("-"," ",text)
    text = re.sub(" +", " ", text)
    text=text.strip('._ ')
    text=text.split(" ")
    highfre_org=['of', 'and', 'department', 'laboratory', 'university', 'engineering', 'institute', 'key', 'school', 'college', 'technology', 'science', 'chemistry', 'for', 'center', 'hospital', 'state', 'chemical', 'medical','materials', 'physics', 'research' ,'chinese', 'sciences', 'information', 'academy', 'surgery', 'medicine', 'china', 'national', 'composite' ,'the', 'ministry', 'lab', 'general', 'first', 'experimental', 'affiliated', 'applied',  'resources','life', 'matter', 'structures' ,'acoustics' ,'oral', 'co', 'advanced', 'normal', 'graduate' ,'ltd', 'prevention','traditional' ,'union','univ', 'an','high']
    text_temp=copy.deepcopy(text)
    for i in text_temp:
        for j in highfre_org:
            if i==j:
                text.remove(i)
    text=' '.join(text)
    return text


# preprocess title / input is the authors'title  "str" / output is the title "str" after preproce
# title预处理：去掉关干扰符号以及高频词
def title_prepro(text):

    text = text.lower()
    text = re.sub("\.", " ", text)
    text = re.sub("\,", " ", text)
    text = re.sub("\'", " ", text)
    text = re.sub("_", " ", text)
    text = re.sub("-", " ", text)
    text = re.sub(" +", " ", text)
    text = text.strip('._ ')
    text = text.split(" ")
    highfre_org = []
    text_temp = copy.deepcopy(text)
    for i in text_temp:
        for j in highfre_org:
            if i == j:
                text.remove(i)
    text = ' '.join(text)
    return text