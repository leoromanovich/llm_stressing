NAME := ollama_stress

run_ollama_stress:
	locust -f main.py \
	--host http://localhost:11434 \
	--endpoint /v1/chat/completions \
	--model qwen3:30b-a3b-instruct-2507-q4_K_M \
	--input-type chat \
	--max-tokens 32 \
	--temperature 0 \
	--stream \
	--headless \
	--csv ollama_stress --csv-full-history \
	--html ollama_stress.html \
	--name ollama_stress \
	--stop-timeout 180


run_stress:
	locust -f main.py \
		--host http://localhost:11434 \
		--model qwen3:30b-a3b-instruct-2507-q4_K_M \
		--endpoint /v1/chat/completions \
		--input-type chat \
		--max-tokens 32 \
		--temperature 0 \
		-u 5 -r 2 -t 5m \
		--stop-timeout 180 \
		--headless --csv $(NAME) --html $(NAME).html \
		--name $(NAME)