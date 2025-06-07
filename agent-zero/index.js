const duckdb = require('duckdb');
const path = require('path');

const dbPath = path.resolve('/data/main.db');

async function main() {
    console.log('Agent Zero application started.');
    console.log('Connecting to DuckDB database at', dbPath);

    const db = new duckdb.Database(dbPath);
    const connection = db.connect();

    connection.exec('CREATE TABLE IF NOT EXISTS logs(timestamp DATETIME, message VARCHAR);', (err) => {
        if (err) {
            console.error('Error creating table:', err);
            return;
        }
        console.log('Table "logs" ready.');
        const now = new Date().toISOString();
        connection.run('INSERT INTO logs VALUES (?, ?)', [now, 'Agent-zero started'], (err) => {
            if (err) {
                console.error('Error inserting data:', err);
            } else {
                console.log('Log entry added.');
            }
        });
    });

    console.log('Initialization complete.');
}

main();
