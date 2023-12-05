## =============================================================================
## twitter explorer 
## helpers
## =============================================================================

import pandas as pd
import datetime as dt
from twitterexplorer.constants import *

def string_to_list(string):
    return string.split("|")

def date_to_datetime(d):
    return dt.datetime(d.year,d.month,d.day)

def load_data(path):
    try:
        df = pd.read_csv(path,
                         dtype=twitwi_schema,
                         usecols=cols_to_load,
                         low_memory=False,
                         lineterminator='\n'
                     )          
        df = df.drop_duplicates('id',keep='last')        
    except (pd.errors.ParserError,ValueError) as e:
        df = pd.read_csv(path,
                         dtype=twitwi_schema,
                         usecols=cols_to_load,
                         low_memory=False,
                         lineterminator='\n'
                     )          
        df = df.drop_duplicates('id',keep='last')                
    return df    

def get_edgelist(df,interaction_type):
    """Generate an edgelist from a Twitter CSV data collection.

    Parameters:
    df (pandas dataframe): dataframe containing the tweets in twitwi format
    interaction_type (str): retweet/quote/reply/mention
    
    Returns:
    interactions (pandas df): df reduced to the selected interactions
    tuples (pandas df): source/target/tweetid/timestamp pandas edgelist
    """    

    if interaction_type == 'retweet':
        interactions = df[df['retweeted_id'].notna()]
        tuples = interactions[['user_id','retweeted_user_id','id','timestamp_utc']]
        tuples = tuples.rename(columns={'user_id':'source',
                                        'retweeted_user_id':'target',
                                        'id':'tweetid',
                                        'timestamp_utc':'timestamp'})
    elif interaction_type == 'quote':
        interactions = df[(df['quoted_id'].notna())&(df['quoted_user_id'].notna())]
        tuples = interactions[['user_id','quoted_user_id','id','timestamp_utc']]
        tuples = tuples.rename(columns={'user_id':'source',
                                        'quoted_user_id':'target',
                                        'id':'tweetid',
                                        'timestamp_utc':'timestamp'})
    elif interaction_type == 'reply':
        interactions = df[df['to_userid'].notna()]
        tuples = interactions[['user_id','to_userid','id','timestamp_utc']]
        tuples = tuples.rename(columns={'user_id':'source',
                                        'to_user_id':'target',
                                        'id':'tweetid',
                                        'timestamp_utc':'timestamp'})
    elif interaction_type == 'mention':
        interactions = df[(df['mentioned_ids'].notna())&(df['mentioned_names'].notna())]
        interactions = interactions[(interactions['retweeted_id'].isna())&(interactions['quoted_id'].isna())&(interactions['to_userid'].isna())]
        interactions['mentioned_ids'] = interactions['mentioned_ids'].apply(string_to_list)
        interactions['mentioned_names'] = interactions['mentioned_names'].apply(string_to_list)        
        interactions = interactions.explode(['mentioned_ids','mentioned_names'])
        tuples = interactions[['user_id','mentioned_ids','id','timestamp_utc']]
        tuples = tuples.rename(columns={'user_id':'source',
                                        'mentioned_ids':'target',
                                         'id':'tweetid',
                                         'timestamp_utc':'timestamp'})        
        
    return interactions,tuples

def read_apikeys(path,version):
    credentials = []
    if version == "v1":
        lines = []
        with open (path,"r",encoding="utf-8") as f:
            for idx,line in enumerate(f.readlines()):
                lines.append(line)
            if lines[0].replace("\n","") != "# api_key" and lines[2].replace("\n","") != "# api_secret_key":
                return False
            else:
                return [lines[1].replace("\n",""),lines[3].replace("\n","")]
    elif version == "v2":
        lines = []
        with open (path,"r",encoding="utf-8") as f:
            for idx,line in enumerate(f.readlines()):
                lines.append(line)
            if lines[0].replace("\n","") != "# bearer token":
                return False
            else:
                return lines[1].replace("\n","")


def get_lang_full(lang_string):
    languages = {
        'fr': 'French',
        'de': 'German',
        'en': 'English',
        'es': 'Spanish',
        'it': 'Italian',
        'pt': 'Portuguese',
        'ru': 'Russian',
        'zh': 'Chinese',
        'ja': 'Japanese',
        'ar': 'Arabic',
        'ko': 'Korean'
        # Add more as needed
    }

    return languages.get(lang_string, lang_string)

def get_time_range(timestamp_utc):
    min_timestamp = timestamp_utc.min()
    max_timestamp = timestamp_utc.max()

    min_date = pd.to_datetime(min_timestamp, unit='s').date()
    max_date = pd.to_datetime(max_timestamp, unit='s').date()

    time_range = f"{min_date} to {max_date}"
    return time_range


def get_mentioned_names(mentioned_names):
    mentioned_names_splitted = mentioned_names.apply(lambda x: str(x).split('|'))
    mentioned_names = pd.Series([name for name_list in mentioned_names_splitted for name in name_list])
    return mentioned_names

def get_hashtags(hashtags):
    hashtags_splitted = hashtags.apply(lambda x: str(x).split('|'))
    hashtags = pd.Series([hashtag for hashtag_list in hashtags_splitted for hashtag in hashtag_list])
    return hashtags