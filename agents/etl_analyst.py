import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.llm_pick import pick_llm
from utils.etl_tools import ETLTools
from utils.database import DatabaseUtil
from models.schema import AgentSchema, JudgeSchema, ETLAgentSchema
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langchain.tools import tool


# -------------------------------------- ETL Agent -------------------------------------------------#


# decorator to register the function as a tool for the agent 
@tool
def extract_load_tool(url:str,output_folder:str,format:str) -> str:
    """
    This tool extracts the data from the API (url) and loads it into the
    the desired location (output_folder).

    Args:
        url (str): The API endpoint from which to extract data.
        output_folder (str): The folder where the extracted data will be saved.
        format (str): The format in which to save the extracted data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.
    """
    
    etl_obj = ETLTools()
    return etl_obj.extract_load(url,output_folder,format)
@tool
def transform_load_tool(input_file_path:str,output_folder:str,output_format:str,user_question:str) -> str:
    """
    This tool transforms the data from the specified path and loads it into the
    the desired location (output_folder).

    Args:
        input_file_path (str): The path to the file containing the data to be transformed.
        output_folder (str): The folder where the transformed data will be saved.
        output_format (str): The format in which to save the transformed data (csv, json, parquet).
    
    Returns:
        str: A message indicating the success or failure of the operation.
    """
    
    etl_tools = ETLTools()

    top_3_rows=etl_tools.transform_load_content(input_file_path)

    # now we will provide to LLM

    llm = pick_llm("medium")  # Pick the appropriate LLM based on the level of the question


    prompt = f"""
            You are a Python Data Analyst who uses Pandas to analyze data. 
            You need to provide only the Pandas Code that will help to perform the right ETL operations on the data stored in the file : {input_file_path}
            as per the user's question. Do not provide any explanation or comments, only
            the code should be provided. The code should be in a format that can be executed 
            in a Python environment with Pandas installed. 
            Don't write anything else than Pandas Code. \n
            
            Create the Pandas Dataframe from the data stored in the file : {input_file_path} and then 
            write the code to transform and save the data at {output_folder}.
            Here's the user's question: {user_question}\n
            Here's the context of the data you will be analyzing: {top_3_rows}\n

        """
    response = llm.invoke(prompt).content  # Generate the Pandas code using the LLM

    # optional cleaning
    pandas_code = response.strip().strip('```').strip().lstrip('python').strip()

    # Execute the generated Pandas code
    results=etl_tools.execute_code(pandas_code)

    return f"The data is transformed and saved at {output_folder} in {output_format} format. \n\n Pandas Code Executed: \n {pandas_code} \n\n Execution Result: \n {results}"

# toolkit
tools=[extract_load_tool,transform_load_tool]

llm=pick_llm("medium")  # Pick the appropriate LLM based on the level of the question
llm_bind=llm.bind_tools(tools)  # Bind the tools to the LLM

#  inorder to run these tools in a loop, we will use the StateGraph to create a loop that will allow the agent to use the tools multiple times until the final answer is generated.

def llm_node(state:ETLAgentSchema):
    messages=state.messages

    prompt = f"""
                You are a Python Data Analyst who has access to tools that can extract and load, 
                transform and load data. You will be provided with a user's question 
                and you would need to perform the right ETL operations as per the user's question. 
                If the operation is performed then inform the user and end the coversation.
                Here's the chat history: {messages}\n
        """
    final_answer = llm_bind.invoke(prompt) # Generate the final answer using the LLM
    state.messages = state.messages + [final_answer]  # Append the final answer to the messages list
    return state

def tool_node(state:ETLAgentSchema):
    """
    This node is responsible for invoking the appropriate tool based on the user's question and the context provided by the LLM.
    """
    # fill this list with tools messages
    tools_results=[]
    tool_by_name={tool.name:tool for tool in tools}
    tool_calls=state.messages[-1].tool_calls

    for tool_call in tool_calls:
        # fetching the tool name from the tools by name
        tool=tool_by_name.get(tool_call['name'])
        observation=tool.invoke(tool_call['args'])
        tools_results.append(ToolMessage(content=observation, tool_call_id=tool_call['id']))

    state.messages = state.messages + tools_results  # Append the tool results to the messages list
    return state

# Nodes & Edges

etl_analyst_graph = StateGraph(ETLAgentSchema)
etl_analyst_graph.add_node("llm_node",llm_node)
etl_analyst_graph.add_node("tool_node",tool_node)

etl_analyst_graph.add_edge(START,"llm_node")
etl_analyst_graph.add_edge("llm_node","tool_node")

def is_tool_call(state:ETLAgentSchema):
    tool_calls=state.messages[-1].tool_calls

    if tool_calls:
        return "tool_node"
    else:
        return "end"
etl_analyst_graph.add_conditional_edges(
        "llm_node",is_tool_call,
        {
            "tool_node":"tool_node",
            "end":END
        }   
        )
etl_analyst_graph.add_edge("tool_node","llm_node")
etl_analyst=etl_analyst_graph.compile()

if __name__ == "__main__":
    # compile the graph and run the agent
    # Optional
    from IPython.display import display, Image
    # img = Image(etl_analyst.get_graph().draw_mermaid_png())
    img = Image(etl_analyst.get_graph().draw_png())
    with open("etl_analyst_graph.png", "wb") as f:
        f.write(img.data)

    # response = etl_analyst.invoke(
    #     {"messages":[HumanMessage(content="I want to extract the data from the API endpoint 'https://pokeapi.co/api/v2/pokemon' and save it to data/extract folder in the csv folder")]}
    # )

    response = etl_analyst.invoke(
         {"messages":[HumanMessage(content=f"""
            I want to transform the data stored in the 'C:\\Users\\K.Nagaraju\Desktop\\Ola AI Project\\data\\extract\\extracted_data.csv' file 
            and save the transformed data in the 'C:\\Users\\K.Nagaraju\Desktop\\Ola AI Project\\data\\transform' folder in the csv format.
            The transformation should filter the data to show bulbasaur pokemon only.
""")]}
    )    

    print(response)

    # llm_bind=pick_llm("medium").bind_tools(tools)  # Pick the appropriate LLM based on the level of the question and bind the tools to it
    # # this will return tool messages
    # print(llm_bind.invoke("I want to extract data from the API at 'https://api.example.com/data' and save it as a CSV file in the 'data/output'."))