import time
import deepl
from auth import *
from typing import Annotated, List, Dict, Optional, Union

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain_google_community import GoogleSearchAPIWrapper, GoogleTranslateTransformer

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager


@tool
def google_search(query: str) -> str:
    """Search Google for recent results."""
    return GoogleSearchAPIWrapper().run(query)

@tool
def scrape_webpages(urls: Union[List[str], str]) -> str:
    """Tool that uses WebBaseLoader to scrape the provided web pages for detailed information."""
    if isinstance(urls, str):
        try:
            urls = json.loads(urls)
        except json.JSONDecodeError:
            urls = [urls]
    
    if not isinstance(urls, list):
        raise ValueError("URLs must be a list of strings or a JSON string representing a list of URLs")
    
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

@tool
def scrape_references(urls: List[str]) -> str:
    """Tool that uses WebBaseLoader to scrape the references for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"


@tool
def read_document(
    file_name: Annotated[str, "File path to save the document."],
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()
    if start is not None:
        start = 0
    return "\n".join(lines[start:end])


@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"


@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""

    with (WORKING_DIRECTORY / file_name).open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with (WORKING_DIRECTORY / file_name).open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"

@tool
def quillbot_detect_ai_content(text: str, timeout: int = 10) -> str:
    """
    Uses Selenium to navigate to Quillbot AI Content Detector, input text, and extract the AI-generated content percentage.
    
    Args:
    text (str): The text to analyze using the AI Content Detector.
    timeout (int): Maximum time to wait for the elements to appear, in seconds.
    
    Returns:
    The AI content percentage or an error message if something goes wrong.
    """
    
    url = "https://quillbot.com/ai-content-detector"

    # Start the WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})
    
    try:
        # Navigate to the URL
        driver.get(url)
        
        wait = WebDriverWait(driver, timeout)

        time.sleep(1)
        input_element = driver.find_element(By.XPATH, "//div[@data-testid='aidr-input-editor']")
        input_element.send_keys(text)
        
        time.sleep(1)
        send_button = driver.find_element(By.XPATH, "//button[@data-testid='aidr-primary-cta']")
        send_button.click()

        time.sleep(1)
        output_element = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[@data-testid='aidr-ai-score-percentage']")
            )
        )
        return f"Text is AI-generated on {output_element.text}"

    except Exception as e:
       return f"An error occurred during AI generation assessment: {str(e)}"
    
    finally:
        driver.quit()

@tool
def gptzero_detect_ai_content(text: str, timeout: int = 3) -> str:
    """
    Uses Selenium to navigate to GPTZero AI Content Detector, input text, and extract the AI-generated content percentage.
    
    Args:
    text (str): The text to analyze using the AI Content Detector.
    timeout (int): Maximum time to wait for the elements to appear, in seconds.
    
    Returns:
    The AI content percentage or an error message if something goes wrong.
    """
    
    url = "https://gptzero.me"

    # Start the WebDriver
    chrome_options = Options()
    #chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})
    
    try:
        # Navigate to the URL
        driver.get(url)
        
        wait = WebDriverWait(driver, timeout)

        time.sleep(1)
        input_element = driver.find_element(By.XPATH, "//textarea[@placeholder='Paste your text here...']")
        input_element.send_keys(text)
        
        time.sleep(1)
        send_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Check Origin')]")
        driver.execute_script("arguments[0].click();", send_button)

        time.sleep(1)
        output_element = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//h1[@id='ai-scan-results-ai-percentage']")
            )
        )
        return f"Text is AI-generated on {output_element.text}"

    except Exception as e:
       return f"An error occurred during AI generation assessment: {str(e)}"
    
    #finally:
    #    driver.quit()

@tool
def deepl_translate(text: str) -> str:
    """
    Uses Deepl API to navigate to translate input text in Russian
    
    Args:
    text (str): The text to translate.
    
    Returns:
    Translated string or an error message if something goes wrong.
    """

    try:
        # Navigate to the URL
        translator = deepl.Translator(os.environ["DEEPL_API_KEY"])
        translation_result = translator.translate_text(text, target_lang="RU")
        return f"Text is AI-generated on {translation_result.text}"

    except Exception as e:
       return f"An error occurred during translation with deepl: {str(e)}"

@tool
def google_translate(text: str, timeout: int = 10) -> str:
    """
    Uses Selenium to navigate to translate text using Google Translate, but without API.
    
    Args:
    text (str): The text to translate.
    timeout (int): Maximum time to wait for the elements to appear, in seconds.
    
    Returns:
    Translated text or an error message if something goes wrong.
    """
    
    source_lang = 'en'
    dest_lang = 'ru'

    # Start the WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--headless")

    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options = chrome_options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.execute_cdp_cmd('Network.setUserAgentOverride', {"userAgent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.53 Safari/537.36'})
    
    try:
        url = "https://translate.google.com.my/?sl="+source_lang+"&tl="+dest_lang+"&text="+text+"&op=translate"
        driver.get(url)
        
        wait = WebDriverWait(driver, timeout)

        # Given below x path contains the translated output that we are storing in output variable:=>
        output_element = wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//span[@class='HwtZe']")
            )
        )
        return f"Translated text: {output_element.text}"

    except Exception as e:
       return f"An error occurred during translation with google: {str(e)}"
    
    finally:
        driver.quit()





