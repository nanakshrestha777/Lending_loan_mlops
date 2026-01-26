#!/bin/bash

# Start MariaDB in background using the original entrypoint
/usr/local/bin/docker-entrypoint.sh mariadbd &
MARIADB_PID=$!

# Wait for MariaDB to be ready
echo "Waiting for MariaDB to be ready..."
for i in {1..60}; do
    if mariadb-admin ping -uroot -p"${MARIADB_ROOT_PASSWORD}" --silent 2>/dev/null; then
        echo "MariaDB is ready!"
        break
    fi
    sleep 2
done

# Provision Columnstore if not already done
if [ ! -f /var/lib/columnstore/.provisioned ]; then
    echo "Provisioning Columnstore..."
    provision mcs1
    if [ $? -eq 0 ]; then
        touch /var/lib/columnstore/.provisioned
        echo "Columnstore provisioned successfully!"
    fi
else
    echo "Columnstore already provisioned."
fi

# Wait for MariaDB process
wait $MARIADB_PID
