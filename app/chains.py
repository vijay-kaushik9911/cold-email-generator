import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama3-8b-8192")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills` and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links):
        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DESCRIPTION:
            {job_description}
            
            ### INSTRUCTION:
            You are Vijay Kaushik, a passionate Full-Stack Developer specializing in building scalable, efficient, and modern web and mobile applications.

            Your technical expertise spans frontend frameworks like React and Next, backend technologies such as Node.js, Express.js, 
            and mobile platforms including Kotlin. You are also proficient in cloud platforms like Firebase and Azure, 
            and DevOps tools such as Docker. You also have knowledge on ML and AI.

            Your task is to write a personalized cold email to the client based on the job description above.

            In the email:
            - Start with the subject and salutations.
            - Briefly explain how your skills align with the job requirements.
            - Emphasize how you can contribute value to their team or project.
            - Select and include the **most relevant portfolio links** from the list provided: {link_list}

            End the email professionally with:
            - Your full name
            - Your designation: Full-Stack Developer
            - Your email: vijaykaushik9911@gmail.com

            Do not include any labels or preamble — just the email body.

            ### EMAIL (NO PREAMBLE):
            """
        )


        chain_email = prompt_email | self.llm
        res = chain_email.invoke({"job_description": str(job), "link_list": links})
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))