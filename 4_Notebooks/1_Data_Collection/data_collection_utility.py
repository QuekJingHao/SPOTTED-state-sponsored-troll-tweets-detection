import numpy as np
import pandas as pd
import shutil as sh
import twint
import nest_asyncio
import snscrape.modules.twitter as sntwitter

def Twint_Scrapper(scrape_limit, verified_users, to_df, to_csv, hide_output):
    
    nest_asyncio.apply()
    twint_scrapper = twint.Config()
    
    for user in verified_users:
        print('Collecting Tweets on', user, '...')
        twint_scrapper.Username = user
        twint_scrapper.Limit = scrape_limit
        #twint_scrapper.Since = '2010-01-01 00:00:00'
        #twint_scrapper.Until = '2022-01-01 00:00:00'
        twint_scrapper.Year = 2022 # get tweets BEFORE 2022
        twint_scrapper.Store_csv = to_csv
        twint_scrapper.Pandas = to_df
        twint_scrapper.Output = user + '.csv'
        twint_scrapper.Hide_output = hide_output
        
        twint.run.Search(twint_scrapper)
        
        Tweets_df = twint.storage.panda.Tweets_df
        print('Length of dataframe:', len(Tweets_df))
        print('[*]--------------------------------------      SUCCESS      --------------------------------------[*]\n')

    print('\n[*] Collection Successful. Total number of scrapped users:', len(verified_users))
    return None
    

# this function takes any number of csv files and concatenate them into a single dataframe
def dataset_fusion(path, files):
    
    df_all = [pd.read_csv(path + file, low_memory = False) for file in files]
    
    df_len_total = 0
    for i in range(len(files)):
        df_len_i = len(df_all[i])
        df_len_total += df_len_i
        print('Length of {} dataframe is {}'.format(files[i], df_len_i))
    
    df_unity = pd.concat(df_all, ignore_index = True)
    
    print('Length of merged dataframe is {}, [{}]'.format(len(df_unity), len(df_unity) == df_len_total))
    
    print('[*]--------------------------------------      SUCCESS      --------------------------------------[*]\n')
    
    return df_unity


def move_verified_datasets(files, strata_name):
    
    current_path = 'D:/STARMAKER Data Collection/Notebooks/'
    verified_path = 'D:/STARMAKER Data Collection/Data/Verified/'
    
    current_file_path = [current_path + file for file in files]
    destinated_file_path = verified_path + strata_name + '/'
    
    print('Moving files to designated folder...')
    for file in current_file_path:
        sh.move(file, destinated_file_path)
    
    print('[*]--------------------------------------      COMPLETE      --------------------------------------[*]\n')
    
    pass


def remove_df_duplicates(df):
    
    df_copy = df.copy()
    print('Dropping duplicates... Initial length of dataframe: {}'.format(len(df_copy)))
    df_copy = df_copy.drop_duplicates(subset = ['tweet_text'])
    print('Removal Completed. Final length of dataframe: {}'.format(len(df_copy)))
          
    return df_copy



# Alternatively, if Twint fails, we can use another tool called snscrape, with the module snscrape.modules.twitter that can use to scrape tweets as well
def SNS_Twitter_Scrapper(limit, user_name):
    
    tweets_list = []
    source = "Twitter"
    
    print('Collecting Tweets on', user_name, ',scrape limit:', limit)
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper('from:' + user_name).get_items()):
        if i >= limit:
            break
        else:
            tweets_list.append([tweet.id, 
                                tweet.url,
                                tweet.user.username,
                                tweet.content,
                                tweet.date,
                                source,
                                tweet.retweetCount,
                                tweet.likeCount,
                                tweet.replyCount])

    tweets_df = pd.DataFrame(tweets_list,
                              columns = ['Tweet_ID', 
                                         'URL', 
                                         "Account_Name", 
                                         'Text', 
                                         'Datetime',
                                         'Source',
                                         'Number_Retweets', 
                                         'Number_Likes', 
                                         'Number_Replies'])
    # write the dataframe as csv file
    tweets_df.to_csv(user_name + '.csv')
    print('[*]--------------------------------------      SUCCESS      --------------------------------------[*]')
    return None