import pandas as pd
import numpy as np
import MeCab
import re
from gensim import corpora, matutils, models
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def main():
    df = pd.read_excel("input/sample.xlsx")
    # IDと単語の紐付け
    corpus_dic,corpus_list,dic = create_corpus_list(df,"自己紹介")
    # TF-IDFの計算
    tfidf_df = create_tfidf_df(corpus_list,corpus_dic,dic)
    # Word_cloud保存
    create_word_cloud(tfidf_df,corpus_dic,"/Users/yoshiki.matsubara/Library/Fonts/Arial Unicode.ttf","materials/wordcloud")
    # LDAのdataframeの計算
    lda_df = create_lda_df(corpus_list,5,dic)


def create_corpus_list(df,columns_name):
    tags = ["名詞","形容詞","形容動詞"]
    mecab = MeCab.Tagger("-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    # 単語リストの作成
    temp = []
    for idx,row in df.iterrows():
        temp_list = []
        parsed = mecab.parse(str(row[columns_name]))
        for s in parsed.splitlines():
            if s != "EOS":
                s = s.split("\t")
                if s[1].split(',')[0] in tags:
                    temp_list.append(s[0])
        temp.append(temp_list)
    # 単語IDの付与
    corpus_dic = corpora.Dictionary(temp)
    # コーパスリストに変換
    corpus_list = [corpus_dic.doc2bow(word_in_text) for word_in_text in temp]
    # IDと単語の紐付け
    dic = corpus_dic.token2id
    dic = {v:k for k,v in dic.items()}
    # TF-IDFの計算
    return corpus_dic,corpus_list,dic


def create_tfidf_df(corpus_list,corpus_dic,dic):
    tfidf_model = models.TfidfModel(corpus_list,normalize=True)
    # tf-idfの計算
    corpus_list_tfidf = tfidf_model[corpus_list]
    # データフレーム作成
    tfidf_df = pd.DataFrame(columns=["WORD","SCORE","DOC_ID"])
    for i in range(corpus_dic.num_docs):
        tfidf_df = tfidf_df.append(pd.DataFrame(corpus_list_tfidf[i],columns=["WORD","SCORE"]).assign(DOC_ID=i))

    # IDを単語に変換
    tfidf_df["WORD"] = tfidf_df["WORD"].replace(dic)
    return tfidf_df


def create_word_cloud(tfidf_df,corpus_dic,fpath,fname):
    fig = plt.figure(figsize=(10,10))
    plt.subplots_adjust(wspace=0.3,hspace=0.3)
    for g in range(corpus_dic.num_docs):
        d = tfidf_df[tfidf_df["DOC_ID"]==g]
        word_freq_dic = {}
        for i,v in d.iterrows():
            word_freq_dic[v["WORD"]] = v["SCORE"]
        ax = fig.add_subplot(4,4,g+1)
        wordcloud = WordCloud(background_color='white',
                              font_path=fpath,
                              width=800,height=600)\
            .generate_from_frequencies(word_freq_dic)
        ax.set_title(g+1)
        ax.tick_params(labelbottom="off",bottom="off") # x軸の削除
        ax.tick_params(labelleft="off",left="off") # y軸の削除
        plt.imshow(wordcloud)
        plt.savefig(fname,dpi=400,pad_inches=0.2,bbox_inches="tight")


def create_lda_df(corpus_list_tfidf,num_topics,dic):
    lda = models.ldamodel.LdaModel(corpus=corpus_list_tfidf,num_topics=num_topics)
    topic_top_tag = []
    for topic in lda.show_topics(-1,formatted=False):
        topic_top_tag.append([token_[0] for token_ in topic[1]])
    df = pd.DataFrame(topic_top_tag).astype(int).replace(dic)
    return df


if __name__ == "__main__":
    main()
