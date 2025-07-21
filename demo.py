import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dotenv import load_dotenv
load_dotenv()

import aqxle
from datasets.load_data import load_tables_from_excel, generate_table_metadata, format_metadata_for_prompt

df = load_tables_from_excel('datasets/W48Tables_Cleaned.xlsx')
metadata = generate_table_metadata(df)
formatted_metadata = format_metadata_for_prompt(metadata)

prompt_path = "src/prompts/code_generator.txt"
with open(prompt_path, "r", encoding="utf-8") as f:
    prompt_template = f.read()

question = input("Ask a question about your data: ")

prompt_with_metadata = prompt_template.replace("{{TABLE_METADATA}}", formatted_metadata).replace("{{question}}", question)

aqxle.init("config.yml")

with aqxle.params(name="logicgen", history_length=5, max_retries=3, logging=True) as session:
    result = (
        session
        .llm("codegen_llm", message=prompt_with_metadata)
        .segment(kernel="python")
        .execute(kernel="python", function="main", df=df)
    )

print(result.data.output)
print(result.data.error)
