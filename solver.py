import os
import time
import argparse
from tqdm import tqdm
from enum import Enum
from typing import Any
from playwright.sync_api import sync_playwright
from openai import OpenAI
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel

LLM_MODEL = "gpt-5.4"

class QuestionType(Enum):
    ESSAY = "essay"
    SHORTANSWER = "shortanswer"
    TRUEFALSE = "truefalse"

class TrueFalseAnswer(BaseModel):
    reason: str
    answer: bool

def main(url: str) -> None:
    client = OpenAI()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto("https://plms.postech.ac.kr/")

        plms_username = os.getenv("PLMS_USERNAME")
        plms_password = os.getenv("PLMS_PASSWORD")

        if page.url.endswith("/login/index.php"):
            page.click("a.btn-login-type-sso")
            page.wait_for_load_state("networkidle")

            page.fill("input[name='login_id']", plms_username)
            page.fill("input[name='login_pwd']", plms_password)
            page.click("input[type='button'][value='LOGIN'][onclick='loginProc()']")
            page.wait_for_load_state("networkidle")

            print("Login successful.")
        else:
            print("Already logged in.")

        page.goto(url)
        page.wait_for_load_state("networkidle")

        response_form = page.query_selector("#responseform")
        if not response_form:
            print("Error: Could not find the response form on the page.")
            browser.close()
            return
        
        container_div = response_form.query_selector("div")
        if not container_div:
            print("Error: Could not find the container div inside the response form.")
            browser.close()
            return

        question_queue = []

        questions = container_div.query_selector_all("div.que")
        if not questions:
            print("Error: Could not find any questions on the page.")
            browser.close()
            return
        
        pbar = tqdm(questions, desc="Processing questions", unit="question")

        for index, question in enumerate(questions):
            question_type = question.get_attribute("class")

            qtext_element = question.query_selector(".qtext")
            if not qtext_element:
                print(f"Warning: Could not find the question text for question {index + 1}. Skipping.")
                continue

            mathjax_elements = qtext_element.query_selector_all(".MathJax")
            for mathjax in mathjax_elements:
                mathjax.evaluate("element => element.remove()")
            
            math_scripts = qtext_element.query_selector_all("script[type='math/tex']")
            for script in math_scripts:
                math_content = script.inner_text().strip()
                script.evaluate(f"(element) => element.replaceWith('${math_content}$')")

            question_text = qtext_element.inner_text().strip()

            if "essay" in question_type:
                question_queue.append((index, question_text, QuestionType.ESSAY))
            elif "shortanswer" in question_type:
                question_queue.append((index, question_text, QuestionType.SHORTANSWER))
            elif "truefalse" in question_type:
                question_queue.append((index, question_text, QuestionType.TRUEFALSE))
            else:
                print(f"Warning: Unrecognized question type for question {index + 1}. Skipping.")

            pbar.update(1)
        
        pbar.close()

        def solve_essay(question_text: str) -> str:
            prompt = f"Please provide a detailed answer to the following question. Reply in Korean. Do not include any emojis, special characters, or formatting. Just provide answer in plain text.\n\nQuestion: {question_text}"
            response = client.responses.create(
                model=LLM_MODEL,
                input=prompt
            )
            return response.output_text
        
        def solve_shortanswer(question_text: str) -> str:
            prompt = f"Fill in the blank for the following question. Reply in Korean. Do not include any emojis, special characters, or formatting. Just provide answer in plain text.\n\nQuestion: {question_text}"
            response = client.responses.create(
                model=LLM_MODEL,
                input=prompt
            )
            return response.output_text
        
        def solve_truefalse(question_text: str) -> TrueFalseAnswer:
            prompt = f"Determine whether the following statement is true or false.\n\nStatement: {question_text}"
            response = client.responses.parse(
                model=LLM_MODEL,
                input=prompt,
                text_format=TrueFalseAnswer
            )
            return response.output_parsed
        
        def solve_question(question_data: tuple[int, str, QuestionType]) -> tuple[int, QuestionType, Any]:
            index, question_text, qtype = question_data

            if qtype == QuestionType.ESSAY:
                return index, qtype, solve_essay(question_text)
            if qtype == QuestionType.SHORTANSWER:
                return index, qtype, solve_shortanswer(question_text)
            if qtype == QuestionType.TRUEFALSE:
                return index, qtype, solve_truefalse(question_text)

            raise ValueError(f"Unrecognized question type for question {index + 1}")

        solved_answers: dict[int, tuple[QuestionType, Any]] = {}

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {executor.submit(solve_question, q): q for q in question_queue}
            for future in as_completed(futures):
                question_data = futures[future]
                qindex = question_data[0]
                try:
                    solved_index, solved_type, solved_answer = future.result()
                    solved_answers[solved_index] = (solved_type, solved_answer)
                except Exception as e:
                    print(f"Error solving question {qindex + 1}: {e}")

        pbar = tqdm(solved_answers.items(), desc="Filling answers", unit="question")

        for index, qtype_and_answer in solved_answers.items():
            qtype, answer = qtype_and_answer
            question_element = questions[index]

            if qtype == QuestionType.ESSAY:
                answer_element = question_element.query_selector("div[contenteditable='true']")
                if answer_element:
                    answer_element.evaluate("(element, value) => { element.innerText = value; }", answer)

            elif qtype == QuestionType.SHORTANSWER:
                answer_element = question_element.query_selector("input[type='text']")
                if answer_element:
                    answer_element.fill(answer)

            elif qtype == QuestionType.TRUEFALSE:
                if answer.answer:
                    true_radio = question_element.query_selector("input[type='radio'][value='1']")
                    if true_radio:
                        true_radio.check()
                else:
                    false_radio = question_element.query_selector("input[type='radio'][value='0']")
                    if false_radio:
                        false_radio.check()
            else:
                print(f"Warning: Unrecognized question type for question {index + 1}. Skipping filling answer.")

            pbar.update(1)
        
        pbar.close()

        print("All questions have been processed.")
        
        while True:
            time.sleep(1)
        
if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="PLMS Solver")
    parser.add_argument("--url", type=str, required=True, help="URL of the PLMS quiz")
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY is not set in the environment variables.")
        exit(1)

    if not os.getenv("PLMS_USERNAME") or not os.getenv("PLMS_PASSWORD"):
        print("Error: PLMS_USERNAME and PLMS_PASSWORD must be set in the environment variables.")
        exit(1)

    main(args.url)