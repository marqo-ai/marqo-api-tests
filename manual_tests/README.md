# Manual API tests dev guide

This is for those tests that are tricky to automate, until we get around to automating them. 

## Set up

Some of manual tests will require credentials. 

Create a `.env` file in this dir to store necessary env vars.


```
# .env

my_cred_1 = <PUT YOUR CRED HERE>
```

## Guide
- Be careful with including `test` into the name of tests methods/ classes, as `pytest` might try run them during automated tests.