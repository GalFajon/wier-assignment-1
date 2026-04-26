# PA2
## How to use:

1. Create an environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

2. Start the database:

   ```powershell
   docker compose up -d db
   ```

3. Import your PA1 crawler database dump manually by downloading the 4thcrawl.sql file
```powershell
docker cp .\4thcrawl.sql crawler-postgres:/tmp/4thcrawl.dump
docker exec -it crawler-postgres pg_restore -U crawler -d crawler --no-owner /tmp/4thcrawl.dump
```

4. Apply PA2 migration changes:

   ```powershell
   docker compose --profile init up migrate
   ```

5. Run parser (XPath + regex stub over html_content):

   ```powershell
   docker compose --profile parse up parser
   ```