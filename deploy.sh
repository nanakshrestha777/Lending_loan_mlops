#!/bin/bash

# MLflow Deployment Helper Script
# Quick commands for deploying models with MLflow

set -e

echo "🚀 MLflow Deployment Helper"
echo "=============================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check if docker-compose is running
check_services() {
    echo "Checking services..."
    
    if ! docker ps | grep -q mlflow_server; then
        print_error "MLflow server is not running"
        echo "Start services with: docker-compose up -d"
        exit 1
    fi
    
    if ! docker ps | grep -q airflow_webserver; then
        print_warning "Airflow webserver is not running"
    fi
    
    print_success "Required services are running"
}

# Deploy model
deploy_model() {
    echo ""
    echo "📦 Deploying model to production..."
    docker exec -it airflow_webserver python /opt/airflow/src/deploy_model.py
    
    if [ $? -eq 0 ]; then
        print_success "Model deployed successfully!"
    else
        print_error "Model deployment failed"
        exit 1
    fi
}

# Reload API
reload_api() {
    echo ""
    echo "🔄 Reloading production API..."
    
    response=$(curl -s -X POST http://localhost:8001/model/reload)
    
    if [ $? -eq 0 ]; then
        print_success "API reload initiated"
        echo "$response"
    else
        print_error "Failed to reload API"
    fi
}

# Check model status
check_model() {
    echo ""
    echo "📊 Checking model status..."
    echo ""
    
    echo "MLflow Models:"
    curl -s http://localhost:5000/api/2.0/mlflow/registered-models/list | python3 -m json.tool || print_error "Failed to fetch models"
    
    echo ""
    echo "Production API Status:"
    curl -s http://localhost:8001/model/info | python3 -m json.tool || print_error "API not responding"
}

# Test prediction
test_prediction() {
    echo ""
    echo "🧪 Testing prediction endpoint..."
    
    curl -X POST http://localhost:8001/predict \
      -H "Content-Type: application/json" \
      -d '{
        "loan_amnt": 10000.0,
        "term": 36,
        "int_rate": 10.5,
        "installment": 325.0,
        "grade": "B",
        "emp_length": 5.0,
        "home_ownership": "RENT",
        "annual_inc": 50000.0,
        "verification_status": "Verified",
        "purpose": "debt_consolidation",
        "dti": 15.5,
        "delinq_2yrs": 0,
        "inq_last_6mths": 1,
        "open_acc": 8,
        "pub_rec": 0,
        "revol_bal": 5000.0,
        "revol_util": 45.0,
        "total_acc": 15
      }' | python3 -m json.tool
    
    echo ""
}

# View stats
view_stats() {
    echo ""
    echo "📈 API Statistics:"
    curl -s http://localhost:8001/stats | python3 -m json.tool
    echo ""
}

# Full deployment workflow
full_deployment() {
    print_success "Starting full deployment workflow..."
    
    check_services
    deploy_model
    sleep 2
    reload_api
    sleep 2
    check_model
    test_prediction
    view_stats
    
    print_success "Deployment complete! 🎉"
}

# Main menu
show_menu() {
    echo ""
    echo "Choose an option:"
    echo "1) Full Deployment (deploy + reload + test)"
    echo "2) Deploy Model Only"
    echo "3) Reload API"
    echo "4) Check Model Status"
    echo "5) Test Prediction"
    echo "6) View Statistics"
    echo "7) Check Services"
    echo "8) Open MLflow UI"
    echo "9) Open Production API Docs"
    echo "0) Exit"
    echo ""
    read -p "Enter option [0-9]: " option
    
    case $option in
        1) full_deployment ;;
        2) deploy_model ;;
        3) reload_api ;;
        4) check_model ;;
        5) test_prediction ;;
        6) view_stats ;;
        7) check_services ;;
        8) 
            print_success "Opening MLflow UI..."
            xdg-open http://localhost:5000 2>/dev/null || open http://localhost:5000 2>/dev/null || echo "Open http://localhost:5000 in your browser"
            ;;
        9) 
            print_success "Opening API Docs..."
            xdg-open http://localhost:8001/docs 2>/dev/null || open http://localhost:8001/docs 2>/dev/null || echo "Open http://localhost:8001/docs in your browser"
            ;;
        0) 
            echo "Goodbye!"
            exit 0
            ;;
        *) 
            print_error "Invalid option"
            ;;
    esac
}

# Run script
if [ "$1" == "deploy" ]; then
    check_services
    deploy_model
elif [ "$1" == "full" ]; then
    full_deployment
elif [ "$1" == "reload" ]; then
    reload_api
elif [ "$1" == "status" ]; then
    check_model
elif [ "$1" == "test" ]; then
    test_prediction
elif [ "$1" == "stats" ]; then
    view_stats
else
    # Interactive menu
    while true; do
        show_menu
    done
fi
