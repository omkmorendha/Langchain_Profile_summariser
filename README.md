# Langchain Profile Summariser
A Langchain application that uses OpenAI, SerpAPI, Nubela API to find information and make inferences about someone using only their name.

APIs used:
1. OpenAI 
2. Nubela
3. SerpAI

We first input a name to a web-application that I created using Flask, which is then passed to the SerpAPI that scrapes the Internet to find a LinkedIn profile page associated with that name,
the profile page is than sent to Nubela API that converts this into a JSON, which is then passed to OpenAI API that gives us some inferences about the person such as a short summary, some interesting facts,
what could be a potential ice-breaker with that person and finally their interests.

All of the above steps are carried out by a Langchain agent.

## Set-up
1. First clone this repository using the command:
   ```
   git clone https://github.com/omkmorendha/Langchain_Profile_summariser
   ```
2. Create a vitrual enviornment using
   ```
   pipenv shell
   ```
   
3. Then create an enviornmental variable file(.env) and link it

4. Add your API keys for the following APIS:
   1. OpenAI API
   2. SerpAPI
   3. Nebula API

5. Run the app.py file

## Final Result
![Picture-1](https://github.com/omkmorendha/Langchain_LinkedIn_summariser/assets/17925053/e2689f56-dbed-4d71-9c34-a77889158d74)

![picture-2](https://github.com/omkmorendha/Langchain_LinkedIn_summariser/assets/17925053/2f18fbd1-d9bf-4e34-871a-43710932db2c)

![picture-3](https://github.com/omkmorendha/Langchain_LinkedIn_summariser/assets/17925053/9556aa3a-fbf3-4d5a-83e1-d497c5230ea3)
