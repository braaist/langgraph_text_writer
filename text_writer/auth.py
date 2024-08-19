import os
from pathlib import Path
from typing_extensions import TypedDict
from tempfile import TemporaryDirectory

#TMP dir
_TEMP_DIRECTORY = TemporaryDirectory()
WORKING_DIRECTORY = Path(_TEMP_DIRECTORY.name)

#API keys
os.environ["GOOGLE_CSE_ID"] = ""
os.environ["GOOGLE_API_KEY"] = ""
os.environ["OPENAI_API_KEY"] = ""
os.environ["ANTROPIC_API_KEY"] = ""
os.environ["DEEPL_API_KEY"] = ""

#Reference links:
reference_links = []