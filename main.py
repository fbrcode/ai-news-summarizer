from openai import OpenAI
import os
from dotenv import find_dotenv, load_dotenv
import time
# import logging
# from datetime import datetime
import requests
import json
import streamlit as st

load_dotenv()
# openai.api_key = os.environ.get("OPENAI_API_KEY")
# defaults to getting the key using os.environ.get("OPENAI_API_KEY")
# if you saved the key under a different environment variable name, you can do something like:
# client = OpenAI(api_key=os.environ.get("CUSTOM_ENV_NAME"))

news_api_key = os.environ.get("NEWS_API_KEY")

client = OpenAI()
model = "gpt-4o-mini"
assistant_id_file = "./assistant.id"

def get_news(topic):
    url = (f"https://newsapi.org/v2/everything?q={topic}&apiKey={news_api_key}&pageSize=5")

    try:
        response = requests.get(url)
        if response.status_code == 200:
            news = json.dumps(response.json(), indent=4)
            news_json = json.loads(news)

            data = news_json

            # Access all the fields == loop through
            status = data["status"]
            total_results = data["totalResults"]
            articles = data["articles"]
            final_news = []

            # Loop through articles
            for article in articles:
                source_name = article["source"]["name"]
                author = article["author"]
                title = article["title"]
                description = article["description"]
                url = article["url"]
                # content = article["content"]
                title_description = ""
                title_description += f"""Title: {title},""" + "\n"
                title_description += f"""Author: {author},""" + "\n"
                title_description += f"""Source: {source_name},""" + "\n"
                title_description += f"""Description: {description},""" + "\n"
                title_description += f"""URL: {url}""" + "\n\n"
                final_news.append(title_description)

            return final_news
        else:
            return []

    except requests.exceptions.RequestException as e:
        print("Error occurred during API Request", e)

def save_assistant_id(id):
    with open(assistant_id_file, "w") as f:
        f.write(id)

def read_assistant_id():
    try:
      with open(assistant_id_file, "r") as f:
          return f.read()
    except FileNotFoundError:
        return None

class AssistantManager:
    # get the assistant id from file, if any
    assistant_id = read_assistant_id()

    def __init__(self, model: str = model):
        self.client = client # openai.OpenAI()
        self.model = model # "gpt-4o-mini"
        self.assistant = None
        self.thread = None
        self.run = None
        self.summary = None

        # retrieve existing assistant if id is already set
        if AssistantManager.assistant_id:
            self.assistant = self.client.beta.assistants.retrieve(assistant_id=AssistantManager.assistant_id)

    def create_assistant(self, name, instructions, tools):
        if not self.assistant:
            assistant_obj = self.client.beta.assistants.create(
                name=name, 
                instructions=instructions, 
                tools=tools, 
                model=self.model
            )
            AssistantManager.assistant_id = assistant_obj.id
            self.assistant = assistant_obj
            # save the assistant id to use it later (avoid creating new assistants with same functionality)
            save_assistant_id(self.assistant.id)
            print(f"Created new Assistant with ID: {self.assistant.id}")
        else:
            print(f"Assistant already exists with ID: {self.assistant.id}")

    def create_thread(self):
        if not self.thread:
            thread_obj = self.client.beta.threads.create()
            AssistantManager.thread_id = thread_obj.id
            self.thread = thread_obj
            print(f"Created new thread with ID: {self.thread.id}")
        else:
            print(f"Thread already exists with ID: {self.thread.id}")

    def add_message_to_thread(self, role, content):
        if self.thread:
            self.client.beta.threads.messages.create(
                thread_id=self.thread.id, 
                role=role, 
                content=content
            )

    def run_assistant(self, instructions):
        if self.assistant and self.thread:
            self.run = self.client.beta.threads.runs.create(
                assistant_id=self.assistant.id,
                thread_id=self.thread.id,
                instructions=instructions,
            )

    def process_message(self):
        if self.thread:
            messages = self.client.beta.threads.messages.list(thread_id=self.thread.id)
            summary = []

            # last message means response from the assistant
            last_message = messages.data[0]
            role = last_message.role
            response = last_message.content[0].text.value
            summary.append(response)

            self.summary = "\n".join(summary)

            print(f"SUMMARY from {role.capitalize()}:\n\n{response}")

            # for msg in messages:
            #     role = msg.role
            #     content = msg.content[0].text.value
            #     print(f"SUMMARY from {role.capitalize()}:\n\n{content}")

    def call_required_functions(self, required_actions):
        if not self.run:
            return
        tool_outputs = []

        for action in required_actions["tool_calls"]:
            function_name = action["function"]["name"]
            arguments = json.loads(action["function"]["arguments"])

            if function_name == "get_news":
                output = get_news(topic=arguments["topic"])
                # print(f"STUFF >>>> {output}")
                print("STUFF >>>>")
                final_str = ""
                for item in output:
                    final_str += "".join(item)

                print(final_str)

                tool_outputs.append({"tool_call_id": action["id"], "output": final_str})
            else:
                raise ValueError(f"Unknown function: {function_name}")

        print("Submitting outputs back to the Assistant...")
        self.client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id, 
            run_id=self.run.id, 
            tool_outputs=tool_outputs
        )

    # for streamlit
    def get_summary(self):
        return self.summary

    def wait_for_completion(self):
        if self.thread and self.run:
            while True:
                time.sleep(5)
                run = self.client.beta.threads.runs.retrieve(
                    thread_id=self.thread.id, 
                    run_id=self.run.id
                )
                print(f"RUN STATUS:: {run.model_dump_json(indent=4)}")

                if run.status == "completed":
                    self.process_message()
                    break
                elif run.status == "requires_action":
                    # print("FUNCTION CALLING NOW...")
                    print(f"*** CALLING FUNCTION MODEL: {run.required_action.submit_tool_outputs.model_dump_json(indent=4)}")
                    self.call_required_functions(
                        required_actions=run.required_action.submit_tool_outputs.model_dump()
                    )

    # Run the steps
    def run_steps(self):
        run_steps = self.client.beta.threads.runs.steps.list(
            thread_id=self.thread.id, 
            run_id=self.run.id
        )
        print(f"\nRun-Steps:::")
        for step in run_steps.data:
            # print(f"\nStep ID: {step.id} [{step.created_at}] ({step.status})\n{step.step_details.to_json()}\n")
            print(f"\n{step.model_dump_json(indent=4)}")
        return run_steps.data


def main():
    # news = get_news("bitcoin")
    # print(news[0])

    manager = AssistantManager()

    # Streamlit interface
    st.title("News Summarizer")

    with st.form(key="user_input_form"):
        topic_input = st.text_input("Enter topic:")
        submit_button = st.form_submit_button(label="Run Assistant")

        if submit_button:
            manager.create_assistant(
                name="News Summarizer",
                instructions="You are a personal article summarizer Assistant who knows how to take a list of article's titles and descriptions and then write a short summary of all the news articles",
                # https://platform.openai.com/docs/guides/function-calling?lang=python
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_news",
                            "description": "Get the list of articles/news for the given topic",
                            "strict": True,
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "topic": {
                                        "type": "string",
                                        "description": "The topic for the news, e.g. bitcoin",
                                    }
                                },
                                "required": ["topic"],
                                "additionalProperties": False,
                            },
                        },
                    }
                ],
            )
            manager.create_thread()

            # Add the message and run the assistant
            manager.add_message_to_thread(
                role="user", content=f"Please, summarize the news on this topic: {topic_input}"
            )
            manager.run_assistant(instructions="Summarize the news")

            # Wait for completions and process messages
            manager.wait_for_completion()

            summary = manager.get_summary()

            st.write(summary)

            st.text("Run Steps:")
            st.code(manager.run_steps(), line_numbers=True)


if __name__ == "__main__":
    main()