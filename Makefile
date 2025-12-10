.PHONY: run-api open-docs api install

# ----------------------------------
#         LOCAL SET UP
# ----------------------------------
install:
	@pip install -r requirements.txt

# ----------------------------------
#     RUNNING FAST_API LOCALLY
# ----------------------------------
run-api:
	uvicorn Fast_api.app:app --reload --port 8000

open-docs:
	open http://127.0.0.1:8000/docs

api: run-api open-docs

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------
#streamlit:
#	-@streamlit run app.py # uncomment when implementing UI
