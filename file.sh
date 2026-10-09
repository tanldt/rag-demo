cd rag-demo && python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu  # lighter build for machines without a GPU
.venv/bin/pip install -r requirements.txt
.venv/bin/python rag_demo.py "What is chunking?"
