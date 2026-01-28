#!/bin/bash

# Script to push 50 random loan requests to the API

API_URL="${1:-http://localhost:8000}"
ENDPOINT="$API_URL/predict/batch"

# Generate 50 random loan requests
cat > /tmp/batch_loans.json <<'EOF'
[
  {"grade": "A", "home_ownership": "RENT", "fico_score": 720, "annual_inc": 65000, "int_rate": 7.5, "dti": 15.2},
  {"grade": "B", "home_ownership": "OWN", "fico_score": 680, "annual_inc": 52000, "int_rate": 10.2, "dti": 18.5},
  {"grade": "C", "home_ownership": "RENT", "fico_score": 640, "annual_inc": 48000, "int_rate": 12.8, "dti": 22.1},
  {"grade": "A", "home_ownership": "MORTGAGE", "fico_score": 750, "annual_inc": 85000, "int_rate": 6.9, "dti": 12.5},
  {"grade": "D", "home_ownership": "RENT", "fico_score": 600, "annual_inc": 38000, "int_rate": 15.5, "dti": 28.3},
  {"grade": "B", "home_ownership": "OWN", "fico_score": 695, "annual_inc": 58000, "int_rate": 9.8, "dti": 16.7},
  {"grade": "C", "home_ownership": "MORTGAGE", "fico_score": 655, "annual_inc": 45000, "int_rate": 11.5, "dti": 20.4},
  {"grade": "A", "home_ownership": "RENT", "fico_score": 735, "annual_inc": 72000, "int_rate": 7.2, "dti": 14.1},
  {"grade": "B", "home_ownership": "RENT", "fico_score": 670, "annual_inc": 50000, "int_rate": 10.5, "dti": 19.2},
  {"grade": "D", "home_ownership": "RENT", "fico_score": 590, "annual_inc": 35000, "int_rate": 16.2, "dti": 30.5},
  {"grade": "C", "home_ownership": "OWN", "fico_score": 645, "annual_inc": 46000, "int_rate": 12.1, "dti": 21.8},
  {"grade": "A", "home_ownership": "MORTGAGE", "fico_score": 760, "annual_inc": 90000, "int_rate": 6.5, "dti": 11.2},
  {"grade": "B", "home_ownership": "RENT", "fico_score": 685, "annual_inc": 54000, "int_rate": 10.0, "dti": 17.9},
  {"grade": "C", "home_ownership": "RENT", "fico_score": 635, "annual_inc": 44000, "int_rate": 13.2, "dti": 23.6},
  {"grade": "D", "home_ownership": "OWN", "fico_score": 605, "annual_inc": 40000, "int_rate": 15.0, "dti": 27.5},
  {"grade": "A", "home_ownership": "OWN", "fico_score": 740, "annual_inc": 78000, "int_rate": 7.0, "dti": 13.5},
  {"grade": "B", "home_ownership": "MORTGAGE", "fico_score": 675, "annual_inc": 51000, "int_rate": 10.3, "dti": 18.1},
  {"grade": "C", "home_ownership": "RENT", "fico_score": 650, "annual_inc": 47000, "int_rate": 11.9, "dti": 20.9},
  {"grade": "A", "home_ownership": "RENT", "fico_score": 725, "annual_inc": 68000, "int_rate": 7.8, "dti": 15.8},
  {"grade": "D", "home_ownership": "RENT", "fico_score": 595, "annual_inc": 36000, "int_rate": 15.8, "dti": 29.7},
  {"grade": "B", "home_ownership": "OWN", "fico_score": 690, "annual_inc": 56000, "int_rate": 9.5, "dti": 17.2},
  {"grade": "C", "home_ownership": "MORTGAGE", "fico_score": 660, "annual_inc": 49000, "int_rate": 11.2, "dti": 19.8},
  {"grade": "A", "home_ownership": "MORTGAGE", "fico_score": 755, "annual_inc": 88000, "int_rate": 6.7, "dti": 11.8},
  {"grade": "B", "home_ownership": "RENT", "fico_score": 680, "annual_inc": 53000, "int_rate": 10.1, "dti": 18.9},
  {"grade": "D", "home_ownership": "RENT", "fico_score": 600, "annual_inc": 37000, "int_rate": 16.0, "dti": 28.9},
  {"grade": "C", "home_ownership": "OWN", "fico_score": 640, "annual_inc": 45000, "int_rate": 12.5, "dti": 22.5},
  {"grade": "A", "home_ownership": "RENT", "fico_score": 730, "annual_inc": 70000, "int_rate": 7.3, "dti": 14.5},
  {"grade": "B", "home_ownership": "MORTGAGE", "fico_score": 685, "annual_inc": 55000, "int_rate": 9.9, "dti": 17.5},
  {"grade": "C", "home_ownership": "RENT", "fico_score": 645, "annual_inc": 46000, "int_rate": 12.0, "dti": 21.2},
  {"grade": "D", "home_ownership": "OWN", "fico_score": 610, "annual_inc": 41000, "int_rate": 14.8, "dti": 26.8},
  {"grade": "A", "home_ownership": "OWN", "fico_score": 745, "annual_inc": 80000, "int_rate": 6.8, "dti": 12.9},
  {"grade": "B", "home_ownership": "RENT", "fico_score": 675, "annual_inc": 52000, "int_rate": 10.4, "dti": 18.6},
  {"grade": "C", "home_ownership": "MORTGAGE", "fico_score": 655, "annual_inc": 48000, "int_rate": 11.6, "dti": 20.1},
  {"grade": "A", "home_ownership": "MORTGAGE", "fico_score": 765, "annual_inc": 95000, "int_rate": 6.4, "dti": 10.8},
  {"grade": "D", "home_ownership": "RENT", "fico_score": 590, "annual_inc": 34000, "int_rate": 16.5, "dti": 31.2},
  {"grade": "B", "home_ownership": "OWN", "fico_score": 695, "annual_inc": 59000, "int_rate": 9.6, "dti": 16.5},
  {"grade": "C", "home_ownership": "RENT", "fico_score": 650, "annual_inc": 47000, "int_rate": 11.8, "dti": 21.5},
  {"grade": "A", "home_ownership": "RENT", "fico_score": 720, "annual_inc": 66000, "int_rate": 7.6, "dti": 15.3},
  {"grade": "B", "home_ownership": "MORTGAGE", "fico_score": 680, "annual_inc": 54000, "int_rate": 10.2, "dti": 18.3},
  {"grade": "D", "home_ownership": "RENT", "fico_score": 605, "annual_inc": 39000, "int_rate": 15.2, "dti": 27.9},
  {"grade": "C", "home_ownership": "OWN", "fico_score": 645, "annual_inc": 46000, "int_rate": 12.3, "dti": 22.0},
  {"grade": "A", "home_ownership": "MORTGAGE", "fico_score": 750, "annual_inc": 86000, "int_rate": 6.9, "dti": 12.1},
  {"grade": "B", "home_ownership": "RENT", "fico_score": 690, "annual_inc": 57000, "int_rate": 9.7, "dti": 17.0},
  {"grade": "C", "home_ownership": "MORTGAGE", "fico_score": 660, "annual_inc": 50000, "int_rate": 11.1, "dti": 19.5},
  {"grade": "A", "home_ownership": "OWN", "fico_score": 735, "annual_inc": 74000, "int_rate": 7.1, "dti": 13.8},
  {"grade": "D", "home_ownership": "RENT", "fico_score": 595, "annual_inc": 36000, "int_rate": 15.9, "dti": 30.1},
  {"grade": "B", "home_ownership": "OWN", "fico_score": 685, "annual_inc": 55000, "int_rate": 9.8, "dti": 17.8},
  {"grade": "C", "home_ownership": "RENT", "fico_score": 640, "annual_inc": 45000, "int_rate": 12.6, "dti": 22.8},
  {"grade": "A", "home_ownership": "RENT", "fico_score": 725, "annual_inc": 69000, "int_rate": 7.4, "dti": 14.8},
  {"grade": "B", "home_ownership": "MORTGAGE", "fico_score": 675, "annual_inc": 53000, "int_rate": 10.5, "dti": 19.0}
]
EOF

echo "📤 Sending 50 loan requests to $ENDPOINT ..."

response=$(curl -s -X POST "$ENDPOINT" \
  -H "Content-Type: application/json" \
  -d @/tmp/batch_loans.json)

echo ""
echo "✅ Response:"
echo "$response" | python3 -m json.tool 2>/dev/null || echo "$response"

# Clean up
rm -f /tmp/batch_loans.json

echo ""
echo "📊 Check results at:"
echo "  - Predictions: $API_URL/ui/history"
echo "  - Drift Report: $API_URL/monitor/drift"
