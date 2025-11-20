from google.colab import userdata

#Getting Gemini API key from Google Colab "Secrets" service
api_key = userdata.get('gemini_api_key')

from typing import Annotated,Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """The state of the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from amadeus import Client, ResponseError
import requests

amadeus = Client(
    #Getting Amadeus API keys from Google Colab "Secrets" service
    client_id = userdata.get('amadeus_client_id'),
    client_secret = userdata.get('amadeus_client_secret')
)

class ToolFields0(BaseModel):
    city:str = Field(description="The name of a city in english - for example Athens")
    country:str = Field(description="The expected country code of the country the city is - for example GR for Athens, you should not inform the user about this field.")

@tool("get_iata_of_city", args_schema=ToolFields0)
def get_iata_of_city( city :str, country :str ) -> dict:
  """Retrieves the possible iata codes based on given city name and country code.
     Returns a list dictionary with those codes and the provided information ( city name and so on ).
     You should intelligently complete the expected country code without to ask the user about that."""
  try:
    cities = amadeus.reference_data.locations.cities.get(keyword=city).data
    reducedCityNames = [ i for i in cities if i.get('address').get('countryCode') == country ]
    if len( reducedCityNames ) == 0:
      reducedCityNames = [ i for i in cities ]

    outputDict = {}
    for i in reducedCityNames:
      iata = i.get('iataCode')
      outputDict[ iata ] = {}
      outputDict[ iata ][ 'name' ] = i.get('name')
      for key in i.get('address'):
        outputDict[ iata ][ key ] = i.get('address').get( key )
      for key in i.get('geoCode'):
        outputDict[ iata ][ key ] = i.get('geoCode').get( key )

    return outputDict

  except:
    return {"error": "The city not found (no IATA code)."}

class ToolFields1(BaseModel):
  iataCityCode:str = Field(description="The iata code of a city (3 characters long) - for example ATH for Athens")
  people:int = Field(description="A number of people - for example for a couple the value whould be 2")
  checkInDate:str = Field(description="The check-in date, written as yyyy-mm-dd")
  checkOutDate:str = Field(description="The check-out date, written as yyyy-mm-dd")

@tool("get_best_offer", args_schema=ToolFields1)
def get_best_offer( iataCityCode :str, people :int, checkInDate :str, checkOutDate :str ) -> dict:
  """Retrieves the best offer from a hotel in the target city.
     Returns the best offer in dictionary form, containing the offer's important information."""

  try:
    hotels = amadeus.reference_data.locations.hotels.by_city.get( cityCode = iataCityCode )
    ids = [ hotel.get('hotelId') for hotel in hotels.data ]

    hotelOffers = []
    for i in range( 0, 99, 50 ):
      req = None
      j = 0   # a value used in order to avoid potential infinite loops
      while not req and j < 5:
        j+=1
        req =  amadeus.shopping.hotel_offers_search.get(
        hotelIds= ids[ i : i+50 ], adults = people, checkInDate = checkInDate, checkOutDate = checkOutDate)

      if req:
        hotelOffers += req.data


    bestprice = 1000000000000
    bestOffer = None
    for hotelOffer in hotelOffers:
      for offer in hotelOffer['offers']:
        currentPrice = float(offer['price']['total'] )
        if currentPrice < bestprice:
          bestprice = currentPrice
          bestOffer = {
              'hotelName': hotelOffer['hotel']['name'],
              'offerTotalAccommodationAmount': offer['price']['total'],
              'offerCurrency': offer['price']['currency'],
              'offerRoomDescription': offer['room']['description']['text'],
              'people': people,
              'checkInDate': checkInDate,
              'checkOutDate': checkOutDate
          }

    return bestOffer
  except:
    return {"error": "No Offers Found"}

class ToolFields2(BaseModel):
  wantedCurrency:str = Field(description="The wanted currency in a three letter string - for example USD for US dollars")
  currentCurrency:str = Field(description="The current currency in a three letter string - for example EUR for euro")
  price:float = Field(description="The float price in current currency - for example 11.20 (EUR)")

@tool("get_price_in_wanted_currency", args_schema=ToolFields2)
def get_price_in_wanted_currency( wantedCurrency :str, currentCurrency :str, price :float ) -> dict:
  """Turns the price at current currency to the same price at the wanted currency - for example 11.20 EUR to 11.20 * x USD, where 1.00 EUR == x USD.
    The price is returned in a dictionary."""

  #Getting Exchage API key from Google Colab "Secrets" service
  url = 'https://v6.exchangerate-api.com/v6/' + userdata.get('exchange_api_key') + '/latest/' + currentCurrency
  response = requests.get(url)
  data = response.json()

  if data:
    if wantedCurrency in data['conversion_rates']:
      x = float( data['conversion_rates'][wantedCurrency] )
      return { 'price': str(price * x) }
    else:
      return {"error": "Could Not Estimate Price"}

  else:
    return {"error": "Could Not Estimate Price"}

@tool("about_you", return_direct=True)
def about_you() -> dict:
  """Informs the user about your functionalities and capabilities."""

  return {"agent_services_info": "You can search for the best hotel offers for the city you want based on your travel criteria. You can also get the offer at the currency you prefer. "}

from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash-preview-05-20",
    temperature=1,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    google_api_key=api_key
 )

tools = [ get_iata_of_city, get_best_offer, get_price_in_wanted_currency, about_you ]

model = llm.bind_tools(tools)

from langchain_core.messages import ToolMessage

tools_by_name = {tool.name: tool for tool in tools}

def calling_model(state: AgentState):
    model_response = model.invoke(state["messages"])
    return {"messages": [model_response]}

def calling_tools(state: AgentState):
    results = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_response = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        results.append(
            ToolMessage(
                content=tool_response,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": results}

def continue_or_end(state: AgentState):
    messages = state["messages"]
    if not messages[-1].tool_calls:
        return "end"
    return "continue"

from langgraph.graph import StateGraph, END

workflow = StateGraph(AgentState)

workflow.add_node("thought_llm", calling_model)

workflow.add_node("tool_set",  calling_tools)

workflow.set_entry_point("thought_llm")

workflow.add_conditional_edges(
    "thought_llm",
    continue_or_end,
    {
        "continue": "tool_set",
        "end": END,
    },
)

workflow.add_edge("tool_set", "thought_llm")

graph = workflow.compile()

import gradio as gr

def predict(message,history):
  chat_messages = []
  chat_messages.append(("system", "You are a friendly Hotel Finder Agent. Use the language provided by the user, use most of the available tools at once if needed to complete the user request and do not complain regarding incompatibilities with the provided tools. The tools parameters if they are strings must be given in english. Answer in a smart way providing all the hotel offer info if it is available."))

  for dictionary in history:
      chat_messages.append((dictionary['role'], dictionary['content']))
  chat_messages.append(("user",str(message)))

  inputs = {"messages": chat_messages}

  for state in graph.stream(inputs, stream_mode="values"):
    if "messages" in state:
      last_message = state["messages"][-1]
      output = last_message.content

  return output

iface = gr.ChatInterface(
    predict,
    type="messages",
    save_history=True
)

iface.launch()

