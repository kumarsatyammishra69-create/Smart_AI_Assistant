from langchain.chains import ConversationChain
from chatbot.llm_engine import get_llm
from chatbot.memory import get_memory

def get_chain():
    llm = get_llm()
    memory = get_memory()
    return ConversationChain(llm=llm, memory=memory, verbose=False)