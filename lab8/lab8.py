import os

from langchain import PromptTemplate
from langchain_cohere import ChatCohere
from langchain_core.messages import HumanMessage


def load_text(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text_content = file.read()
        print("File loaded successfully!")
        return text_content
    except Exception as error:
        print("Error loading file:", str(error))
        return None


def build_prompt(text_content):
    template = """
You are an AI assistant helping to summarize and analyze a text document.
Here is the document content:
{text}

Summary:
- Provide a concise summary of the document.

Key Takeaways:
- List 3 important points from the text.

Sentiment Analysis:
- Determine if the sentiment of the document is Positive, Negative, or Neutral.
"""

    prompt_template = PromptTemplate(input_variables=["text"], template=template)
    return prompt_template.format(text=text_content)


def generate_response(formatted_prompt, cohere_api_key, model_name="command-a-03-2025"):
    cohere_llm = ChatCohere(cohere_api_key=cohere_api_key, model=model_name)
    return cohere_llm.invoke([HumanMessage(content=formatted_prompt)]).content


def main():
    file_path = input("Enter the text file path: ").strip()
    cohere_api_key = os.environ.get("COHERE_API_KEY")

    if not cohere_api_key:
        cohere_api_key = input("Enter your Cohere API key: ").strip()

    text_content = load_text(file_path)
    if not text_content:
        return

    formatted_prompt = build_prompt(text_content)
    print("\nFormatted Prompt:\n")
    print(formatted_prompt)

    response = generate_response(formatted_prompt, cohere_api_key)
    print("\nFormatted Output:\n")
    print(response)


if __name__ == "__main__":
    main()