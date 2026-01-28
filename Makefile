# Configuration
ENV_NAME=cmp6230_test
CONDA_CMD=conda
YAML_FILE=environment.yml
API_DIR=src
API_MAIN=api_main
API_HOST=0.0.0.0
API_PORT=8000

# Default: Start everything
.PHONY: all
all:
	@echo "Starting MLOps Environment..."
	@echo ""
	@$(CONDA_CMD) env list | grep -q "^$(ENV_NAME)\s" || \
		(echo "📦 Creating environment $(ENV_NAME)..." && $(CONDA_CMD) env create -f $(YAML_FILE))
	@echo "✅ Environment ready: $(ENV_NAME)"
	@echo ""
	@echo "🐳 Starting Docker services..."
	@docker compose up -d
	@echo ""
	@echo "📊 Starting Jupyter Notebook..."
	@pgrep -f "jupyter-notebook" > /dev/null || \
		nohup $(CONDA_CMD) run -n $(ENV_NAME) jupyter notebook --ip="*" --port=8888 --no-browser --allow-root > jupyter.log 2>&1 & 
	@sleep 2
	@echo ""
	@echo "⚡ Starting FastAPI..."
	@pgrep -f "uvicorn $(API_MAIN):app" > /dev/null || \
		(cd $(API_DIR) && nohup $(CONDA_CMD) run -n $(ENV_NAME) \
		env REDIS_HOST=127.0.0.1 POSTGRES_HOST=127.0.0.1 MLFLOW_TRACKING_URI=http://127.0.0.1:5000 \
		uvicorn $(API_MAIN):app --host $(API_HOST) --port $(API_PORT) > ../api.log 2>&1 &)
	@sleep 2
	@echo ""
	@echo "========================================="
	@echo "🚀 ALL SERVICES RUNNING!"
	@echo "========================================="
	@$(CONDA_CMD) run -n $(ENV_NAME) jupyter notebook list 2>/dev/null || echo "Jupyter: check jupyter.log"
	@echo "FastAPI: http://localhost:$(API_PORT)/docs"
	@echo "Environment: conda activate $(ENV_NAME)"
	@echo ""

# Check status
.PHONY: status
status:
	@echo "========================================="
	@echo "📊 STATUS"
	@echo "========================================="
	@$(CONDA_CMD) env list | grep -q "^$(ENV_NAME)\s" && echo "✅ Environment: $(ENV_NAME)" || echo "❌ Environment not created"
	@pgrep -f "jupyter-notebook" > /dev/null && echo "✅ Jupyter: Running" || echo "❌ Jupyter: Not running"
	@pgrep -f "uvicorn $(API_MAIN):app" > /dev/null && echo "✅ FastAPI: Running" || echo "❌ FastAPI: Not running"
	@docker compose ps 2>/dev/null | grep -q "Up" && echo "✅ Docker: Running" || echo "❌ Docker: Not running"
	@echo ""

# Stop services
.PHONY: stop
stop:
	@echo "Stopping services..."
	@pkill -f "jupyter-notebook" 2>/dev/null || true
	@pkill -f "uvicorn $(API_MAIN):app" 2>/dev/null || true
	@docker compose down 2>/dev/null || true
	@rm -f jupyter.log api.log jupyter.pid api.pid 2>/dev/null || true
	@echo "✅ Services stopped"

# Delete environment
.PHONY: delete clean
delete:
	@$(MAKE) -s stop
	@echo "Deleting environment $(ENV_NAME)..."
	@$(CONDA_CMD) env remove --name $(ENV_NAME) -y 2>/dev/null && echo "✅ Environment deleted" || echo "Environment not found"

clean: delete

