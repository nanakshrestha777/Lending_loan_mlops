#!/bin/bash
set -e

echo "🚀 Starting MariaDB ColumnStore initialization..."

# The container's default entrypoint handles MariaDB startup
# We just need to ensure provisioning happens
if [ ! -f /var/lib/columnstore/.provisioned ]; then
    echo "⏳ Waiting for MariaDB to be ready before provisioning..."
    sleep 15
    
    echo "🔧 Provisioning ColumnStore cluster..."
    provision mcs1
    
    if [ $? -eq 0 ]; then
        touch /var/lib/columnstore/.provisioned
        echo "✅ ColumnStore provisioned successfully!"
    else
        echo "❌ ColumnStore provisioning failed!"
        exit 1
    fi
else
    echo "✅ ColumnStore already provisioned, skipping..."
fi

echo "🎉 ColumnStore ready for use!"
