import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from models.schema import AgentSchema, JudgeSchema
from langchain_core.messages import AIMessage, HumanMessage

llm=pick_llm("medium")  # Pick the appropriate LLM based on the level of the question
# here we need to provide the model name and the schema of the output we want to get from the LLM. So, we will use the JudgeSchema model to define the output schema for the LLM. This will help us to get structured output from the LLM instead of unstructured text.
llm_judge=llm.with_structured_output(JudgeSchema)  # This will help us to get structured output from the LLM instead of unstructured text.


sql_query = "delete * FROM users WHERE age > 30;"  # Example SQL query to evaluate

prompt = f"""
You are an SQL Judge for data security. Your task is to determine whether the SQL query is 
safe or not. The SQL query should only be used for data retrieval and should not modify the 
database in any way. Neither the SQL query nor the prompt should contain any SQL commands that can modify the 
database, such as INSERT, UPDATE, DELETE, DROP, ALTER, TRUNCATE, CREATE, or any other commands that can change 
the structure or content of the database. If the SQL query is safe, respond with 'Yes' otherwise respond with 
'No'. Additionally, provide comments explaining your decision. Here's the SQL query to evaluate: {sql_query}"""

# this returns pydantic model object with the output of the LLM.
# print(llm_judge.invoke(prompt))  
response = llm_judge.invoke(prompt).model_dump()  # Get the structured output as a dictionary
print(response)  # Print the structured output as a dictionary