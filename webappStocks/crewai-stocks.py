# Libs imports
import json
import os
import datetime

import yfinance as yf

from crewai import Agent, Task, Crew
from crewai.process import Process

from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults

import streamlit as st

# In[40]:


# Create Yahoo Finance Tool


def fetch_stock_price(ticket: str):
    stock = yf.download(ticket, start="2023-08-08", end="2024-08-08")

    return stock


yahoo_finance_tool = Tool(
    name="Yahoo Finance Tool",
    description="Fetches stocks prices for {ticket} from the last year about a specific company from Yahoo Finance API",
    func=lambda ticket: fetch_stock_price(ticket),
)


# In[41]:


# Import OpenAI LLM - GPT

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
)


# In[42]:


# Create agent

stockPriceAnalyst = Agent(
    role="Sênior Stock Price Analyst",
    goal="Find the {ticket} stock price and analyses trends",
    backstory=""" 
  You're highly experienced stock price analyst with a deep understanding of the stock market. 
  You have been analyzing stock prices for years and have a proven track record of making accurate predictions. 
  Your goal is to find the stock price of a specific company and analyze trends to help investors make informed decisions. 
  You have access to historical stock price data and can use this information to make predictions about future stock prices. You are confident in your abilities and are always looking for new opportunities to use your expertise to help others.""",
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=False,
    tools=[yahoo_finance_tool],
)


# In[43]:


# Create task

getStockPrice = Task(
    description="Analyze the stock {ticket} price history and create a trend analyses of up, down or sideways",
    expected_output="""
  Specify the current trend stock price - up, down or sideways

  eg. stock='AAPL, price UP'
  """,
    agent=stockPriceAnalyst,
)


# In[44]:


# Important a tool to search news about a specific company

search_tool = DuckDuckGoSearchResults(backend="news", num_results=10)


# In[45]:


newsAnalyst = Agent(
    role="Stock News Analyst",
    goal="""
  Create a short summary of the market news related to the stock {ticket} company. 
  Specify the current trend - up, down or sideways with the news context. For each request stock asset,
  specity a number between 0 and 100, where 0 is extreme fear and 100 is extreme greed.
  """,
    backstory=""" 
  You're a senior news analyst with a deep understanding of the news industry. 
  You have been analyzing news for years and have a proven track record of making accurate predictions. 
  Your goal is to analyze the news and provide insights to help investors make informed decisions. 
  You have access to news sources and can use this information to make predictions about future events. You are confident in your abilities and are always looking for new opportunities to use your expertise to help others.""",
    verbose=True,
    llm=llm,
    max_iter=10,
    memory=True,
    allow_delegation=False,
    tools=[search_tool],
)


# In[46]:


get_news = Task(
    description=f"""Take the stock and always include BTC to it (if not requested). 
  Use the search tool to search one individually.

  The current date is {datetime.datetime.now()}.

  Compose the results into a helpfull report.
  """,
    expected_output="""
  A summary of the overall market sentiment and one sentence summary for each request asset.
  Include a fear/greed score for each asset based on the news sentiment. Use format:

  <STOCK ASSET>
  <SUMMARY BASED ON NEWS>
  <TREND PREDICTION>
  <FEAR/GREED SCORE>
  """,
    agent=newsAnalyst,
)


# In[47]:


stockAnalystWriter = Agent(
    role="Stock News Analyst",
    goal="""
  Write an insighfull compelling and informative 3 paragraph long newsletter based on the stock report and price trend.
  """,
    backstory=""" 
  You're widely accepted as the best stock analyst in the market.
  You understand complex concepts and create compelling stories and narratives that resonate witth wider audiences.

  You understand macro factores and combine multiple theories and concepts to create a compelling narrative.
  eg. cycle theory and fundamental analysis. 
  
  You're able to hold multiple opinions and perspectives at once and can switch between them as needed.
  """,
    verbose=True,
    llm=llm,
    max_iter=5,
    memory=True,
    allow_delegation=True,
    tools=[search_tool],
)


# In[48]:


writeAnalyses = Task(
    description="""Use thew stock price trend and the stock news report to create an analyses and write the newsletter about the {ticket} that is brief and highlights the most important points.
    Focus on the stock price, news and fear/greed score. What are the near future considerations?

    Include the previous analyses of stock trend and news summary.
  """,
    expected_output="""
  An eloquent 3 paragraphs newsletter formatted as markdown in an easy readable manner. It should contain:

  - 3 bullets executive summary
  - Introduction - set the overall picture and spike up the interest
  - main part provides the meat of the analysis including the news summary and fear/greed scores
  - summary - key facts and concrete future trend prediction - up, down or sideways.
  """,
    agent=stockAnalystWriter,
    context=[getStockPrice, get_news],
)


# In[54]:


crew = Crew(
    agents=[stockPriceAnalyst, newsAnalyst, stockAnalystWriter],
    tasks=[getStockPrice, get_news, writeAnalyses],
    process=Process.hierarchical,
    share_crew=False,
    manager_llm=llm,
    verbose=True,
    max_rpm=15,
)


# In[55]:


# results = crew.kickoff(inputs={"ticket": "AAPL"})


with st.sidebar:
    st.header("Enter the ticket stock: ")

    with st.form(key="research_form"):
        topic = st.text_input("Select the ticket")
        submit_button = st.form_submit_button(label="Run research")

if submit_button:
    if not topic:
        st.error("Please enter a ticket stock field")
    else:
        results = crew.kickoff(inputs={"ticket": topic})

        st.subheader("Results of your research: ")
        st.write(results)


# In[59]:
