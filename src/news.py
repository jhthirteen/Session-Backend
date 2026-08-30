import feedparser
import time
import datetime
import os
import json
from typing import Literal, Optional
from pydantic import BaseModel, Field
from groq import Groq

espn_rss_url = 'https://www.espn.com/espn/rss/nba/news'
SYSTEM_PROMPT = """You are an expert sports journalism editor classifying NBA story headlines. Respond in JSON format.

Rules:
1. 'News' = Hard transactions (trades, signings, 2-year deals), official court filings, official injury reports, jersey retirements.
2. 'Opinion' = Panel forecasts, trade speculation/predictions, fan stories, power rankings, previews, recaps.
3. 'Ambiguous' = ONLY use if the headline is genuinely impossible to categorize without clicking the link. Short team slang (e.g. 'Pels', 'Wolves') or transaction shorthand ('2-year, $12.4M deal', '3-and-D wing') MUST be correctly categorized as News.
"""

class ArticleClassification(BaseModel):
    category: Literal["News", "Opinion", "Ambiguous"] = Field(
        description="News: Objective events, verified trades, court filings, official transactions. Opinion: Panel predictions, fan reactions, forecasts, debate topics."
    )
    reasoning: Optional[str] = Field(
        default="No reasoning provided.",
        description="Brief 1-sentence justification."
    )

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

def classify_headline(title: str, client: Groq) -> ArticleClassification:
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Classify this headline: '{title}'"}
        ],
        response_format={"type": "json_object"},
        temperature=0.0
    )
    
    raw_json = response.choices[0].message.content
    data = json.loads(raw_json)
    return ArticleClassification(**data)


def filter_news_feed_for_stories(stories: list) -> None:
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    for story in stories:
        result = classify_headline(story.title, client)
        print(f'Story: {story.title} is classified as {result.category} with the following rationale: {result.reasoning}')

filter_news_feed_for_stories(fetch_espn_rss_news_past_day())

