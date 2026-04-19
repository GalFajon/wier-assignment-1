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

3. Import your PA1 crawler database dump manually (as you planned).

4. Apply PA2 migration changes:

   ```powershell
   docker compose --profile init up migrate
   ```

5. Run parser (XPath + regex stub over html_content):

   ```powershell
   docker compose --profile parse up parser
   ```