from dotenv import load_dotenv
from langchain_ollama import ChatOllama

llama = ChatOllama(
    model="llama3:8b",
    temperature=0,
    base_url="http://localhost:11434",
    num_predict=256)

if __name__ == "__main__":
    response = llama.invoke("Hello. How are you doing?")
    print(response.content)

