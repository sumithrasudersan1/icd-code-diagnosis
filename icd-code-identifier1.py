from openai import OpenAI
import os
import dotenv
import glob
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = "sk-proj-"+os.getenv("OPENAI_API_KEY")
client = OpenAI()

def add_files_to_vector(vectorId,file_list):
    file_streams = [open(file,'rb') for file in file_list]
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vectorId,
        files = file_streams
    )
    return file_batch
def get_all_ehr_docs(dir_path):
    file_list = []
    file_paths = glob.glob(os.path.join(dir_path,"**",'*'),recursive=True)
    for file_path in file_paths:
        if os.path.isfile(file_path):
            print("File path ",file_path)
            file_list.append(file_path)
    print("Final file list ",file_list)
    return file_list

def get_annotated_files(annotations):
    citations = []
    list_of_files_cited = []
    for index,annotation in enumerate(annotations):
        print("Annotation Index is ",index)
        if annotation.type == "file_citation":
            print("yes annotation type is file_citation",annotation)
            file_id = annotation.file_citation.file_id
            list_of_files_cited.append(file_id)
    if len(list_of_files_cited) > 0:
        list_of_file_cited = list(set(list_of_files_cited))
        print(f" list of files cited after removing duplicates {list_of_files_cited} of length {len(list_of_files_cited)}")
        for j_index, file_id in enumerate(list_of_files_cited):
            cited_file = client.files.retrieve(file_id)
            citations.append(f"[{j_index}] {cited_file.filename}")
        return list_of_files_cited, citations
    else:
        return None,None
def start_chat(thread_id,assistant_id):
    while True:
        prompt = input("What do you want to search? ")
        if prompt.lower() == "quit":
            break
        message = client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=prompt,
        )
        #into universal medical codes to maintain accurate medical records
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread_id,
            assistant_id=assistant_id,
            instructions="You are a medical coding analyst agent. your job is to analyze patient's medical document \
                          such as physician's notes, lab reports, procedures and diagnoses and match the medical conditions with Section111ValidICD10 \
                          Provide a proper justification of why the ICD code has been selected. \
                          Provide your response in a table format with S.No, ICD code, Reason, Presence in Report \
                          you also need to validate the list of medical codes given by the reviewer and check the same against the document for its availability. \
                          Always provide consistent results. If you are not able to find a valid code, please say i don't know",
            tools=[{"type":"file_search"}]
        )
        messages = list(client.beta.threads.messages.list(thread_id=thread_id,run_id=run.id))
        for index,content in enumerate(messages[0].content):
            if content.type == 'text':
                message_content = content.text
                if message_content:
                        annotations = message_content.annotations
                        citations = []
                        list_of_file_ref,citations = get_annotated_files(annotations)
                        if list_of_file_ref is not None:
                            print("list of files",list_of_file_ref)
                            for index,annotation in enumerate(annotations):
                                print("in for loop ",index)
                                for j_index, file_id in enumerate(list_of_file_ref):
                                    print("in j_index")
                                    print(f"j_index is {j_index} and file id is {file_id}")
                                    print(f"file id from annotation is {annotation.file_citation.file_id}")
                                    if file_id == annotation.file_citation.file_id:
                                        print("file id comparison is true ")
                                        message_content.value = message_content.value.replace(annotation.text,f"[{j_index}]")
                        
                        print(message_content.value)
                        final_response = message_content.value
                        #citations = "\n".join(citations)
                        #print("Citations ...",citations)

def create_thread(icd_file):
    message_file = client.files.create(
        file = open(icd_file,"rb"),purpose="assistants"
    )
    thread = client.beta.threads.create(
        messages=[
            {
                "role":"user",
                "content":"Dummy Text",
                "attachments":[
                    {
                        "file_id":message_file.id,
                        "tools":[
                            {
                                "type" : "file_search"
                            }
                        ]
                    }
                ],
            }
        ]
    )
    return thread.id

def main():

    add_files = False
    #icd_file_ref = "icd-codes/Section111ValidICD10-Jan2024.txt"
    file_ref = input("Refer the file to query ?")
    icd_file_ref = f"sample-files\{file_ref}"
    print(f"Refering the file {icd_file_ref}")
    continue_to_search = True
    try:
        vector_store = client.beta.vector_stores.retrieve(os.getenv("VECTOR_STORE"))
        vector_id = vector_store.id
       
        if add_files:
            files = get_all_ehr_docs("sample-files/")
            file_batch = add_files_to_vector(vector_id,files)
            print(f"file_batch status {file_batch.status} and file_batch file count is {file_batch.file_counts}")
        if continue_to_search:
            try:
                list_of_vectors_to_search = [vector_id]
                thread_id = create_thread(icd_file_ref)
                print("Start chat..")
                assistant = client.beta.assistants.retrieve(os.getenv("ASSISTANT"))
                assistant_id = assistant.id
                start_chat(thread_id,assistant_id)
            except Exception as ex:
                print("Something went wrong while initiating chat...",ex)
    except Exception as ex:
        print("Something went wrong ",ex)



if __name__ == "__main__":
    main()