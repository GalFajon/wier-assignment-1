# PA2

## Retrieval demo setup - automatic setup

1. Create and edit the environment file (recommended to leave same as example file):

   ```powershell
   Copy-Item .env.example .env
   ```

2. Start the saved database (automatically downloads and restores the database from Google Drive):

   ```powershell
   docker compose -f docker-compose.yml -f docker-compose.restored.yml up --build --force-recreate db
   ```

   Wait until database system is ready to accept connections, then detach.

3. Run the retrieval demo

   **Option A - Run with Docker**  
   Runs the demo container interactively and installs all required dependencies automatically. Models are still downloaded to pa2/models.

   ```powershell
   docker compose -f docker-compose.yml -f docker-compose.restored.yml run --rm demo
   ```

   **Option B - Run locally with Python virtual environment**  
   Manually install dependencies and run the client script locally.

   ```powershell
   pip install -r requirements.txt
   ```

   ```powershell
   python .\implementation-extraction\demo.py
   ```
   
   or run the demo client with powershell:

   ```powershell
   .\run_demo_client.ps1
   ```

   



## Full parsing setup

1. Create an environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Start the empty database:

   ```powershell
   docker compose up -d db
   ```

3. Import the PA1 crawler database dump manually by downloading the `4thcrawl.sql` file (configure docker command to match your `.env` file).

   Download database dump:

   ```powershell
   gdown https://drive.google.com/file/d/1b-02LDtQxvTFElDnRrLYQsXa5oWgVKvO/view?usp=drive_link
   ```

   Copy dump into PostgreSQL container and restore:

   ```powershell
   docker cp .\4thcrawl.sql crawler-postgres:/tmp/4thcrawl.dump
   docker exec -it crawler-postgres pg_restore -U crawler -d crawler --no-owner /tmp/4thcrawl.dump
   ```

4. Apply PA2 migration changes:

   ```powershell
   docker compose --profile init up migrate
   ```

5. Run parsing and embedding

   **Option A - Run with Docker (recommended)**  
   Runs parsing and embedding fully inside the container.

   ```powershell
   docker compose --profile parse up parser
   ```

   **Option B - Run locally with Python virtual environment**  
   Install dependencies locally and run the parser manually.

   ```powershell
   pip install -r requirements.txt
   ```

   Run locally with Powershell to pass .env variables:

   ```powershell
    .\run_parser_locally.ps1
   ```