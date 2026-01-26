#!/bin/bash
set -e

PROVISION_FLAG="/var/lib/mysql/provisioned"
if [ ! -f "$PROVISION_FLAG" ]; then
    echo "🚀 Provisioning ColumnStore Engine..."
    provision mcs1
    touch "$PROVISION_FLAG"
else
    echo "✅ ColumnStore already provisioned."
fi

echo "⏳ Waiting for MariaDB to be fully ready..."
until mariadb -uroot -p"${MARIADB_ROOT_PASSWORD}" -e "SELECT 1;" >/dev/null 2>&1; do
    sleep 3
done
echo "✅ MariaDB is ready."

echo "🔑 Ensuring admin user and lending_club DB exist..."
mariadb -uroot -p"${MARIADB_ROOT_PASSWORD}" <<-EOSQL
    -- Drop existing conflicting users to avoid ALTER USER errors
    DROP USER IF EXISTS 'admin'@'%';
    DROP USER IF EXISTS 'admin'@'localhost';

    -- Recreate admin user
    CREATE USER 'admin'@'%' IDENTIFIED BY 'Admin@1234Strong!';
    CREATE USER 'admin'@'localhost' IDENTIFIED BY 'Admin@1234Strong!';
    GRANT ALL PRIVILEGES ON *.* TO 'admin'@'%' WITH GRANT OPTION;
    GRANT ALL PRIVILEGES ON *.* TO 'admin'@'localhost' WITH GRANT OPTION;

    -- Create DB if missing
    CREATE DATABASE IF NOT EXISTS lending_club;

    FLUSH PRIVILEGES;
EOSQL

echo "✅ MariaDB initialization complete!"
