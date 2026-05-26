#!/bin/bash

echo "=== NBA Player Cards Asset Pipeline ==="
echo

# Install dependencies if needed
echo "Checking dependencies..."
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Test the pipeline
echo "Testing pipeline..."
npx ts-node src/asset-pipeline/test-pipeline.ts

echo
echo "=== Pipeline Test Complete ==="
echo "To generate cards, run: npm run build-cards"
echo "To test renderer: npm run render-test"