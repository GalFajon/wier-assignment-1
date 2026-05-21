#!/bin/bash
set -e

ollama serve &
OLLAMA_PID=$!

for i in {1..60}; do
  if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "Ollama is ready!"
    break
  fi
  sleep 1
done

ollama pull llama3.2:1b

wait $OLLAMA_PID
