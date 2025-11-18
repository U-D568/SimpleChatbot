### Usage
1. Enter your OpenAI API Key in left sidebar. If it's not provided, the chat will not open.

2. (optional) upload the documents you want to chat about.
    - Internally, the documents are chunked and top 4 most relevant chunks are retreived when you ask a question.

3. Ask anything you want.


### Installation
1. Create virtual envorionment with python 3.12

2. Install dependencies
```
pip install -r requirements.txt

streamlit run app.py
```

3. Prepare your own ChatGPT API
If API Key is not provided the app will not work.
