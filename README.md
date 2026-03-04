# wier-assignment-1
Assignment 1 webcrawler repo.

## Project structure:
- exercises-notebook contains the notebooks from učilnica, dependencies are in requirements.txt
- src contains source code, current plan is that the crawler will be a docker-compose project:
    - db -> a database with a persistent storage volume
    - server -> rest API in flask
    - client -> container that crawls and updates the API, made to be ran in parallel