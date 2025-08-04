run_stress:
	locust -f main.py \
		--host http://localhost:11434 \
		--model qwen3:30b-a3b-instruct-2507-q4_K_M \
		--endpoint /v1/chat/completions \
		--input-type chat \
		--max-tokens 32 \
		--temperature 0 \
		-u 5 -r 2 -t 2m \
		--stop-timeout 180 \
		--headless --csv vllm_nonstream --html vllm_nonstream.html \
		--name vllm_chat_nonstream