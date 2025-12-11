.PHONY: run-api open-docs api install docker-run

# ----------------------------------
#         LOCAL SET UP
# ----------------------------------
install:
	@pip install -r requirements.txt

# ----------------------------------
#     RUNNING FAST_API LOCALLY
# ----------------------------------
run-api:
	uvicorn Fast_api.app:app --port 8000

open-docs:
	open http://127.0.0.1:8000/docs


api:
	$(MAKE) run-api &
	@sleep 2
	$(MAKE) open-docs

# ----------------------------------
#         HEROKU COMMANDS
# ----------------------------------
#streamlit:
#	-@streamlit run app.py # uncomment when implementing UI

# ----------------------------------
#             DOCKER
# ----------------------------------
# Run the Docker image locally
docker-run:
	docker run -it --rm \
		-p 8000:8000 \
		--env-file .env \
		$(shell docker images "lunar-crater-*:dev" --format "{{.Repository}}:{{.Tag}}" | head -1)
