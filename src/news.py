import feedparser
import time
import datetime

espn_rss_url = 'https://www.espn.com/espn/rss/nba/news'

def fetch_espn_rss_news_past_day() -> list:
    espn_rss_url = 'https://www.espn.com/espn/rss/nba/news'
    current_utc_time = time.gmtime()
    # convert time.struct_type to datetime for timestamp arithmetic 
    current_utc_time = datetime.datetime(*current_utc_time[:6])
    baseline_time = current_utc_time - datetime.timedelta(days=1)
    
    rss_feed = feedparser.parse(espn_rss_url)

    # identify news stories that occurred in the past 24 hour news cycle
    stories_past_day = []
    for story in rss_feed.entries:
        story_time = story.published_parsed
        story_time = datetime.datetime(*story_time[:6])
        if story_time > baseline_time:
            stories_past_day.append(story)
        
    return stories_past_day


def filter_news_feed_for_stories(stories: list) -> None:
    pass

