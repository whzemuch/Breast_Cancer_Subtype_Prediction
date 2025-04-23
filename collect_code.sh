rm -f project_code_collection.zip
zip -r project_model_code_collection.zip \
       callbacks/*.py \
       configs/*.yaml  \
       data/clean_data/*.csv \
       data_loader/*.py \
       losses/*.py  \
       models/*py \
       tests/*.py \
       trainers/*.py \
       utils/*.py \
       tutorials/*.ipynb \
       *.yaml ./*.py ./*.ipynb *.ini *.sh