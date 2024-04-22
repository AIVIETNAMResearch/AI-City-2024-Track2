# Installation

```
pip install -U pip setuptools wheel
pip install -U spacy
python -m spacy download en_core_web_sm
pip install -r src/preprocess/segment_extraction/requirements.txt
pip install flash-attn --no-build-isolation
```

# Set environment variable
```
export HF_TOKEN="put your hf token here"
```

# Run the file
```
python src\preprocess\segment_extraction\sentence_segment.py
```