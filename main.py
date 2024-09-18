from fastapi import FastAPI, File, UploadFile, HTTPException
from PyPDF2 import PdfReader
import openai
import pinecone
import os
from pinecone import Pinecone as PineconeClient
from pinecone import Pinecone, PineconeConfigurationError, ServerlessSpec
from langchain.embeddings import OpenAIEmbeddings
from pdfminer.high_level import extract_text
from langchain.vectorstores import Pinecone
from langchain_community.vectorstores import Pinecone
from pydantic import BaseModel
from typing import List
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQA
import hashlib
import uvicorn

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env",override=True)


app = FastAPI()


PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")

if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")

# Initialize Pinecone with API key

try:
    pc = PineconeClient(api_key=os.environ.get("PINECONE_API_KEY"))
except PineconeConfigurationError as e:
    print(f"Error initializing Pinecone: {e}")

index_name = "pdf-index"

# Create a Pinecone index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')
    )

#index = pinecone.GRPCIndex(index_name)
index = pc.Index(index_name)

embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])



text_field = "text"  # the metadata field that contains our text

# initialize the vector store object
vectorstore = Pinecone(
    pc.Index(index_name), embeddings.embed_query, text_field
)

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=1,
    return_messages=True
)


from PyPDF2 import PdfReader

def extract_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


from PyPDF2 import PdfReader

def extract_text_by_page(file):
    reader = PdfReader(file)
    texts = []
    for i, page in enumerate(reader.pages):
        texts.append({"page": i, "text": page.extract_text()})
    return texts

import uuid

def generate_page_id(page_number):
    return f"page-{page_number}-{uuid.uuid4().hex}"



def generate_embedding(text):
    # Replace this with your actual embedding generation code
    return embeddings.embed_query(text)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    try:
        texts_by_page = extract_text_by_page(file.file)  # Extract text from each page
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading PDF: {str(e)}")

    if not texts_by_page:
        raise HTTPException(status_code=400, detail="No text found in the uploaded PDF.")

    try:
        for page in texts_by_page:
            page_number = page["page"]
            text = page["text"]
            vector = generate_embedding(text)  # Generate vector for the extracted text

            # Generate a unique ID for each page
            page_id = generate_page_id(page_number)

            # Upsert the vector into Pinecone with the unique ID
            index.upsert(
                vectors=[
                    {
                        "id": page_id,
                        "values": vector,
                        "metadata": {"text": text}  # Optional metadata
                    }
                ]
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error inserting vectors into Pinecone: {str(e)}")

    return {"message": "PDF text uploaded and vector inserted successfully."}



greetings = ["hi", "helo", "hello", "hey", "good morning", "good evening","good evening","anyone here",   'howdy','salutations','hiya','hey there','good day',"what's up",'how are you','yo','hi there',"how's it going","how's everything","how's life",'nice to see you','pleased to meet you','good to see you','welcome','hi ya','hello there','how have you been',"what's going on",'how are things','how do you do',"what's happening","how's your day","how's your day going","what's new","what's good","how's it hanging",'how are things going',"how's your morning","how's your afternoon","how's your evening",'what have you been up to',
    'long time no see',"it's been a while",'how have you been lately',"how's everything going","how's it been","how's your week","what's the latest","how's your weekend",'what have you been doing',"what's been happening",'yes','no','sure','yeah','nope','yep','nah','okay','alright','absolutely','of course','definitely','affirmative','negative','indeed','certainly','sure thing','yup','uh huh','no way','not at all','by all means','no thanks','roger','right','fine','okay dokey','okie dokie','for sure','no problem','you bet','absolutely not','no doubt','unquestionably','without a doubt','no chance','yes please','not really','totally',"I'm in",'I agree',"I'm on board",'that works',"I'll pass",'not interested','why not','sure thing','count me in','you got it']
reply= [
    """Welcome to Iotric, what will you like to opt for.
    1. Explore our services.
    2. Explore our Portfolio.
    3. Iotric Products.
    4. Iotric Blogs.
    5. Contact us.
    6. Schedule a meeting."""
]

blog_keywords = ["blog", "blogs", "articles", "iotric blogs", "read", "latest blog", "read blog",
    "blog posts", "recent blogs", "recent articles", "write-ups", "latest articles",
    "blog section", "blog page", "blog content", "published articles", "featured blog",
    "latest write-ups", "blog updates", "new blogs", "new articles", "recent posts",
    "insights", "company blog", "industry articles", "tech blogs", "technology articles",
    "trending blogs", "trending articles", "expert insights", "thought leadership",
    "blogging", "blogging platform", "blog collections", "content hub", "knowledge base",
    "online articles", "digital articles", "company write-ups", "company insights"]
blog_reply = [
    "Here are some of our latest blogs:",
    "1. How to use MVP development to mitigate risk?: https://www.iotric.com/mvp-development-to-mitigate-risk/",
    "2. Minimum Viable Product (MVP) vs Minimum Marketable Product (MMP): https://www.iotric.com/mvp-vs-mmp/",
    "3. What is Fractional Ownership in Real Estate Investment with Blockchain?: https://www.iotric.com/what-is-fractional-ownership-in-real-estate-investment-with-blockchain/",
    "Visit our blog page https://www.iotric.com/blog/ for more articles!"
]



def get_greeting_reply(user_input):
    user_input = user_input.lower()
    if user_input in greetings:
        return reply
    
    if any(keyword in user_input for keyword in blog_keywords):
        return blog_reply

    return None

llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)
# conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=1,
    return_messages=True
)


class MessageRequest(BaseModel):
    message: str
@app.post("/search")
async def print_message(request: MessageRequest):
    try:
        query = request.message
        # Perform similarity search
        search_results = vectorstore.similarity_search(query, k=1)  # Adjust k as needed

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        # Check for greeting or blog reply
        reply = get_greeting_reply(query)
        if reply:
            return {"reply": reply}

        # If not a greeting, perform retrieval-based QA
        modified_query = query + " in points"
        response = qa.invoke(modified_query)
        if 'result' in response:
            return {"result": response["result"]}
        else:
            return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/delete-index/{index_name}")
async def delete_index(index_name: str):
    try:
        indexes = pc.list_indexes()  # Fetch available indexes
        print(f"Available indexes: {indexes}")  # Debug output

        # if index_name not in indexes:
        #     raise HTTPException(status_code=404, detail=f"Index '{index_name}' does not exist.")

        # Attempt to delete the index
        pc.delete_index(index_name)
        return {"message": f"Index '{index_name}' deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting index: {str(e)}")


@app.get("/list-indexes")
async def list_indexes():
    try:
        indexes = pc.list_indexes().names()  # List available indexes
        return {"indexes": indexes}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing indexes: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)

