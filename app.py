## Import necessary libraries/modules
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import JinaEmbeddings
from langchain_core.prompts import PromptTemplate
from typing import List
import load_dotenv
from langchain.text_splitter import MarkdownTextSplitter
from langchain.chains import LLMChain, StuffDocumentsChain
import getpass
import os
from typing import List
import json
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader



#load_dotenv()
#On promt enter groq api key with out quotes
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key:")

# Load environment variables
jina_api=os.getenv('jina_api')#get free jina_api key for embeddings and save it in colabs secret


# Instantiation of LLM model
model = ChatGroq(
    model="llama-3.2-90b-text-preview",
    temperature=0,   # For fetching factual information keep it 0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

prompt_template = """
Context: {context}
Question: {question}

Please extract and organize the following information from the provided veterinary records into a structured health summary,\
Follow the exact format specified below, maintaining all sections even if data is not available for some fields.
Required Format Structure:

1. Client Information
Extract and format the following client details:

Full Name: [Extract complete name]
Phone Number: [Extract all contact numbers]
Complete Address: [Extract full mailing address]
Email Address: [Extract email]

2. Patient Information
Identify and list:

Pet's Name: [Extract full name]
Breed: [Extract specific breed information]
Date of Birth/Age: [Extract DOB or age]
Gender: [Extract gender/sex]
Microchip Number: [Extract complete microchip number if available]

3. Veterinary Clinic Information
For each visit, extract:

Visit Date: [Format as MM/DD/YYYY]
Clinic Name: [Extract complete facility name]
Phone Number: [Extract clinic contact number]
List chronologically, with most recent first.

4. Vaccination Records
For each vaccination, provide:

Vaccine Name: [Extract specific vaccine name]
Administration Date: [Format as MM/DD/YYYY]
Manufacturer Tag Number: [Extract if available]
List chronologically, with most recent first.

5. Medical Information(<Weight History>)
Create a chronological table with:
Weight
Include dates for each measurement.

6. Patient Alerts
List all:

Medical conditions
Allergies
Behavioral notes
Special handling requirements
Format as bullet points.

7. Medications
For each medication, provide:

Medicine Name: [Extract complete drug name]
Strength/Dosage: [Extract concentration/strength]
Quantity: [Extract number prescribed]
Prescription Details: [Extract administration instructions]

8. Laboratory Results
Organize into separate sections:
a) Chemistry
b) CBC (Complete Blood Count)
c) Endocrinology
d) Urinalysis
e) Heartworm Test Results
f) Fecal Results
g) Other Tests
For each test type, create a table with:

Test Marker/Parameter
Results for each date
Include reference ranges where available

9. Imaging Results
Radiographs
For each x-ray:

Date: [Format as MM/DD/YYYY]
Position/View: [Extract specific view]
Results/Findings: [Extract interpretation]

Ultrasound
For each ultrasound:

Date: [Format as MM/DD/YYYY]
Results/Findings: [Extract complete interpretation]

Special Instructions

Table format
create table for every section asked for in correct manner as following\
 <Table Heading>
 |<Data field1>| <Data field2>| <etc>| 

Data Organization:


Maintain chronological order (newest to oldest) for all dated entries
Preserve exact measurements and units as provided
Include "Not Available" for missing information rather than leaving blank


Formatting Requirements:


Use consistent date format throughout (MM/DD/YYYY)
Maintain table structure as shown in template
Use bullet points for lists
Preserve any emphasis (bold, italics) from source documents


Accuracy Checks:


Verify all numerical values
Cross-reference dates across sections for consistency
Ensure proper matching of test results with dates


Quality Guidelines:


Extract complete information without abbreviating
Maintain medical terminology as presented
Preserve specific measurements and units
Include all relevant notes and comments

Please process the provided veterinary records according to these specifications, 
maintaining the exact structure and formatting as shown in the template while ensuring all available information is accurately captured and organized.



Answer: Think step by step and only provide requested information in the requested format.
stop: "```"
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)


def document_parser(filepath):
    # set up parser
    parser = LlamaParse(
        result_type="markdown"  # "markdown" and "text" are available
    )
    # use SimpleDirectoryReader to parse our file
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(input_files=[filepath], file_extractor=file_extractor).load_data()

    # Convert list of Document instances to a single string
    documents_string = "\n".join(str(doc) for doc in documents)  # Convert each Document to string

    with open('/output/output.md', 'w') as f:
        f.write(documents_string)  # Write the string to the file




def get_ans(query: str, docs:List[str]):
    """Get answer using late chunking and contextual retrieval with structured output"""
    try:
        # Split text into chunks of 1000 words with overlap of 200 words for better context
        # Load markdown content from file
        with open('/data/output.md', 'r') as file:
            markdown_content = file.read()

        # Initialize the Markdown Text Splitter
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)

        # Split the content into sections
        split_sections = splitter.split_text(markdown_content)


        # Create embeddings
        embeddings = JinaEmbeddings(
            jina_api_key=jina_api,
            model_name="jina-embeddings-v3"
        )

        # Create vector store for storing embeddinngs with index for rich context retrieval
        docsearch = FAISS.from_texts(
            [doc for doc in split_sections],
            embeddings)

        # Retrieve relevant documents
        """set k(it fetches k number of relevent embeddings for similarity search
        ) value as per your requirement., more the better but may lead to contextual conflict
        choose craefully.
        """
        retrieved_docs = docsearch.similarity_search(query, k=10)

        # Initialize QA chain
        qa_chain = load_qa_chain(
            llm=model,
            chain_type="stuff",
            prompt=PROMPT,
            verbose=False #(on True gives all steps taken by llm )
        )


        # Run the chain and parse the response
        response = qa_chain.run(input_documents=retrieved_docs, question=query)

        return response

    except Exception as e:
        return json.dumps({"error": str(e)})



document_parser("/data/sample_input.pdf")
doc ="/output/output.md"
result = get_ans('provide me output as per {prompt}',doc)
print(result)

