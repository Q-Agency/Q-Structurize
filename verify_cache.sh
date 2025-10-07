#!/bin/bash
# Cache Path Verification Script

echo "=================================="
echo "🔍 Cache Path Verification"
echo "=================================="
echo ""

echo "📂 Checking HOST paths:"
echo ""

# Check host cache
if [ -d "./cache" ]; then
    echo "✅ ./cache exists"
    
    if [ -d "./cache/huggingface" ]; then
        echo "✅ ./cache/huggingface exists"
    else
        echo "⚠️  ./cache/huggingface NOT found"
    fi
    
    if [ -d "./cache/huggingface/hub" ]; then
        echo "✅ ./cache/huggingface/hub exists"
        
        # Check for granite model
        if [ -d "./cache/huggingface/hub/models--ibm-granite--granite-docling-258M" ]; then
            echo "✅ Granite-Docling model found!"
            echo "   📍 ./cache/huggingface/hub/models--ibm-granite--granite-docling-258M/"
            
            # Show size
            size=$(du -sh "./cache/huggingface/hub/models--ibm-granite--granite-docling-258M" 2>/dev/null | cut -f1)
            echo "   📊 Size: $size"
        else
            echo "⚠️  Granite-Docling model NOT found"
            echo "   Expected: ./cache/huggingface/hub/models--ibm-granite--granite-docling-258M/"
        fi
    else
        echo "⚠️  ./cache/huggingface/hub NOT found"
    fi
else
    echo "❌ ./cache directory NOT found"
    echo "   (This is normal before first build)"
fi

echo ""
echo "=================================="
echo "🐳 Checking CONTAINER paths:"
echo "=================================="
echo ""

# Check if container is running
if docker ps --format '{{.Names}}' | grep -q "q-structurize"; then
    echo "✅ Container 'q-structurize' is running"
    echo ""
    
    # Check container paths
    docker exec q-structurize bash -c '
        echo "📂 Container cache structure:"
        echo ""
        
        if [ -d "/app/.cache" ]; then
            echo "✅ /app/.cache exists"
        else
            echo "❌ /app/.cache NOT found"
        fi
        
        if [ -d "/app/.cache/huggingface" ]; then
            echo "✅ /app/.cache/huggingface exists"
        else
            echo "❌ /app/.cache/huggingface NOT found"
        fi
        
        if [ -d "/app/.cache/huggingface/hub" ]; then
            echo "✅ /app/.cache/huggingface/hub exists"
        else
            echo "❌ /app/.cache/huggingface/hub NOT found"
        fi
        
        echo ""
        echo "🔍 Searching for Granite-Docling model:"
        if [ -d "/app/.cache/huggingface/hub/models--ibm-granite--granite-docling-258M" ]; then
            echo "✅ FOUND: /app/.cache/huggingface/hub/models--ibm-granite--granite-docling-258M/"
            du -sh "/app/.cache/huggingface/hub/models--ibm-granite--granite-docling-258M" 2>/dev/null
        else
            echo "⚠️  Model not in expected location"
            echo "   Searching for granite models..."
            find /app/.cache -type d -name "*granite*" 2>/dev/null | head -5
        fi
        
        echo ""
        echo "📊 Environment variables:"
        echo "   HF_HOME=$HF_HOME"
        echo "   HF_HUB_CACHE=$HF_HUB_CACHE"
        echo "   TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
    '
else
    echo "⚠️  Container 'q-structurize' is not running"
    echo "   Start it with: docker-compose -f docker-compose.gpu.yml up -d"
fi

echo ""
echo "=================================="
echo "✅ Verification Complete"
echo "=================================="

