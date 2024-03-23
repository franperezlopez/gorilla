# Based on code from https://github.com/ShishirPatil/gorilla/blob/main/raft/raft.py
import argparse
import random
from typing import List, Optional

import pandas as pd
import PyPDF2
from datasets import Dataset
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.chat_models import ChatOpenAI
from tqdm import tqdm

DEFAULT_CHUNK_SIZE = 512
DEFAULT_QUESTIONS = 5
DEFAULT_DISTRACTORS = 3
DEFAULT_ORACLE_PERCENTAGE = 1.0
DEFAULT_MINIMUM_CHUNK_SIZE = 40

def get_args() -> any:
    """
    Parses and returns the arguments specified by the user's command
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", type=str, required=True, help="The path at which the document is located")
    parser.add_argument("--output_folder", type=str, default="./", help="The folder at which to save the dataset")
    parser.add_argument("--distractors", type=int, default=DEFAULT_DISTRACTORS, help="The number of distractor documents to include per data point / triplet")
    parser.add_argument("--p", type=float, default=DEFAULT_ORACLE_PERCENTAGE, help="The percentage that the oracle document is included in the context")
    parser.add_argument("--questions", type=int, default=DEFAULT_QUESTIONS, help="The number of data points / triplets to generate per chunk")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, help="The size of each chunk in number of tokens")
    parser.add_argument("--max_chunks", type=int, default=None, help="The maximum number of chunks to process")

    args = parser.parse_args()
    return args

def get_chunks(llm_embedding, file_path: str, chunk_size=DEFAULT_CHUNK_SIZE) -> list[str]:
    """
    Takes in a `file_path` and `doctype`, retrieves the document, breaks it down into chunks of size
    `chunk_size`, and returns the chunks.
    """
    chunks = []
    
    text = ""
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
        
    num_chunks = len(text) / chunk_size 
    text_splitter = SemanticChunker(llm_embedding, number_of_chunks=num_chunks)
    chunks = text_splitter.create_documents([text])
    chunks = [chunk.page_content for chunk in chunks]
            
    return chunks

def generate_instructions(chunk, x=5, llm_critic:ChatOpenAI=None) -> list[str]:
    """
    Generates `x` questions / use cases for `chunk`. Used when the input document is of general types 
    `pdf`, `json`, or `txt`.
    """
    class QuestionAnswer(BaseModel):
        question: str = Field(..., description="The question to be answered")
        answer: str = Field(..., description="The answer to the question")
    
    class GeneratorResponse(BaseModel):
        questions: List[QuestionAnswer] = Field(..., description="Pairs of questions and answers generated from the given context.")

    SYSTEM_PROMPT = """You are a synthetic question-answer pair generator. 
Given a chunk of context about some topic(s), generate {number_of_questions} example questions a user could ask and would be answered using information from the chunk. 
For example, if the given context was a Wikipedia paragraph about the United States, an example question could be: 'How many states are in the United States?'"""

    model_with_structure = llm_critic.with_structured_output(GeneratorResponse)
    
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), 
                                               ("system", "The questions should be able to be answered in a few words or less."), 
                                               ("user", "{chunk}")])
    chain = prompt | model_with_structure
    questions: GeneratorResponse = chain.invoke({"chunk":chunk, "number_of_questions":x})

    return [q.dict() for q in questions.questions]

def encode_question() -> list[str]:
    """
    Encode multiple prompt instructions into a single string for the general case (`pdf`, `json`, or `txt`).
    """
            
    prompt = """
        Question: {question}\nContext: {context}\n
        Answer this question using the information given in the context above. Here is things to pay attention to: 
        - First provide step-by-step reasoning on how to answer the question. 
        - In the reasoning, if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. 
        - End your response with final answer in the form <ANSWER>: $answer, the answer should be succint.
    """
    return ChatPromptTemplate.from_messages([("system", "You are a helpful question answerer who can provide an answer given a question and relevant context."), 
                                             ("system", prompt)])

def generate_label(question, context,  llm_critic:ChatOpenAI=None) -> str:
    """
    Generates the label / answer to `question` using `context` and GPT-4.
    """
    class CriticAnswer(BaseModel):
        reasoning: str = Field(..., description="step-by-step reasoning on how to answer the question. if you need to copy paste some sentences from the context, include them in ##begin_quote## and ##end_quote##. This would mean that things outside of ##begin_quote## and ##end_quote## are not directly copy paste from the context. ")
        answer: str = Field(..., description="the final answer to the question, should be succint.")

    model_with_structure = llm_critic.with_structured_output(CriticAnswer)
    prompt = ChatPromptTemplate.from_messages([("system", "You are a helpful question answerer who can provide an answer given a question and relevant context."), 
                                               ("system", "Answer the user question using the information given in the context."), 
                                               ("user", "Question: {question}\nContext: {context}\n")])

    chain = prompt | model_with_structure
    response: CriticAnswer = chain.invoke({"question":question, "context":context})
    return response.dict()

def add_chunk_to_dataset(llm_critic, chunks: list, chunk: str, num_questions: int = 5, num_distract: int = 3, p: float = 1.0) -> list:
    """
    Given a chunk, create {Q, A, D} triplets and add them to the dataset.
    """
    ds = []
    i = chunks.index(chunk)
    qa_pairs = generate_instructions(chunk, num_questions, llm_critic)
    for pair in qa_pairs:
        # add num_distract distractor docs
        (indices := list(range(len(chunks)))).remove(i)
        docs = [chunk] + [chunks[idx] for idx in random.sample(indices, num_distract)]

        # decides whether to add oracle document
        oracle = random.uniform(0, 1) < p
        if not oracle:
            docs[0] = chunks[random.sample(indices, 1)[0]]
        random.shuffle(docs)

        # add answer to q
        cot = generate_label(pair, chunk, llm_critic)

        datapt = {
            "type": "general",
            "question": pair["question"],
            "answer": pair["answer"],
            "context": docs,
            "oracle_context": chunk,
            "cot": cot
        }

        # add to dataset
        ds.append(datapt)
        
    return ds

def enrich_dataset(ds):
    for i,d in enumerate(ds):
        d["id"] = f"seed_task_{i}"
        d["prompt_instruction"] = d["question"]
        d["prompt_input"] = "\n".join([f"<DOCUMENT>{doc}</DOCUMENT>" for doc in d["context"]])
        d["prompt_response"] = d["answer"]
        d["prompt_cot_response"] = f"<SCRATCHPAD>{d['cot']['reasoning']}</SCRATCHPAD>\n<ANSWER>{d['cot']['answer']}</ANSWER>"
    return ds

def main(llm_critic, llm_emb, file_path: str, output: str,
         chunk_size: int = DEFAULT_CHUNK_SIZE, num_questions: int = DEFAULT_QUESTIONS, 
         distractors: int = DEFAULT_DISTRACTORS, p: float = DEFAULT_ORACLE_PERCENTAGE,
         max_chunks: Optional[int] = None):
    """
    Generates a dataset based on the given parameters.

    Args:
        llm_critic (ChatOpenAI): The language model to use for generating answers.
        llm_emb (OpenAIEmbeddings): The language model to use for generating embeddings.
        file_path (str): The path to the data file.
        chunk_size (int): The size of each chunk.
        questions (int): The number of questions to generate for each chunk.
        distractors (int): The number of distractors to include for each question.
        output (str): The output directory to save the generated dataset.
        max_chunks (Optional[int]): The maximum number of chunks to process. If None, all chunks are processed.

    Returns:
        None
    """

    print("Generating chunks...")
    chunks = get_chunks(llm_emb, file_path, chunk_size)
    chunks = list(filter(lambda x: len(x) > DEFAULT_MINIMUM_CHUNK_SIZE, chunks))

    ds = []

    print("Generating dataset...")
    chunks_to_process = random.sample(chunks, max_chunks) if max_chunks is not None and len(chunks) > max_chunks else chunks
    for chunk in tqdm(chunks_to_process):
        try:
            ds += add_chunk_to_dataset(llm_critic=llm_critic, chunks=chunks, chunk=chunk, num_questions=num_questions, p=p,
                                       num_distract=distractors)
        except Exception as e:
            print(e)
            continue

    ds = enrich_dataset(ds)
    print("Dataset generated!")    
    print(f"Dataset length: {len(ds)}")
    if ds is None or len(ds) == 0:
        return

    # print (ds[0])
    ds = Dataset.from_pandas(pd.DataFrame(data=ds))

    # Save as .arrow format
    ds.save_to_disk(output)

    # Save as .jsonl format
    ds.to_json(output + "rag.jsonl")

if __name__ == "__main__":
    # get arguments
    args = get_args()

    llm_critic = ChatOpenAI(model="gpt-4-0125-preview")
    llm_emb = OpenAIEmbeddings(model="text-embedding-3-small")

    # run program
    main(llm_critic=llm_critic, llm_emb=llm_emb, file_path=args.input_file, output=args.output_folder, 
         chunk_size=args.chunk_size, num_questions=args.questions, distractors=args.distractors, 
         p=args.p, max_chunks=args.max_chunks)
