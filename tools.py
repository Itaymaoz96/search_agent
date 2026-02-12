from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(data: str, file_name: str = "research_output.txt"):

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    formatted_text = f"# Research Output - \n{timestamp}\n\n{data}\n\n"
    try:
        with open(file_name, "a", encoding="utf-8") as f:
            f.write(formatted_text)
        return f"Saved to {file_name}"
    except Exception as e:
        return f"Error saving to {file_name}: {e}"

api_wrapper = WikipediaAPIWrapper(top_k_results=1,lang="en",doc_content_chars_max=100)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)
search_tool = Tool(
    name="search",
    func=wiki.run,
    description="Search Wikipedia for information. Use for factual, historical, and general knowledge queries.",
)

save_tool = Tool(
    name="save_text_file",
    func=save_to_txt,
    description="Save the research output to a text file.",
)
