I setup a venv for the test heres what i ran for pip/conda


conda install -c conda-forge pip setuptools wheel -y
conda install -c conda-forge "spacy==3.5.4" "gensim>=4.3,<5" -y
pip install "coreferee==1.4.1"
pip install "pydantic<1.11,!=1.8,!=1.8.1"
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.5.0/en_core_web_trf-3.5.0.tar.gz
python -m coreferee install en
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.5.0/en_core_web_lg-3.5.0.tar.gz

