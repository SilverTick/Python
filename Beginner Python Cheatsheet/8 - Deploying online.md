# Setting up your Git (in Python)
A step-by-step guide to setting up Git repositories, and deploying to Heroku, using Python and Visual Studio Code! It was really painful for me the first few times I did it (with the help of my expert friend, no less), so I really hope the below will help you guys simplify it.

For clarification on the below instructions, when I say “Input `xyzzy`“ I mean type `xyz` into the terminal and hit return/enter. 

__In Visual Studio Code:__
1. Open the terminal. Drag up the bottom console if you can't see it, alternatively click 'View', click 'Integrated Terminal'.
2. Input `cd` - this brings you back to your Home folder
3. Input `ls` - this shows you all the files in the current directory you are in
4. Input `mkdir filename` - this creates a folder with the filename, which is going to be where all your repository files will be in. This filename should be similar to the name of your directory.
5. Input `cd filename` - replace filename with the name of your folder. This brings you into the folder

In Github:
1. Signup for a free account/ login
2. Click ‘New repository’
3. Fill in the repository name, select ‘add README.md’
4. Click ‘Create repository’

In Visual Studio Code:
1. Input ‘git init’ - initialises a git repository locally in your computer
2. Input ‘git add README.md’ - adds the readme file into the repository index. *If you didn’t select add README previously, it will return an error. You will need to create a README file manually by inputing ‘touch README.md’, and then repeat step 2.
3. Input ‘git commit -m “first commit” ’ - basically this commits (aka mark as confirmed) your action (adding the readme file). The text in between the quotes “” describes the action
4. Input ‘git remote add origin https://github.com/username/reponame.git' - this tells git which address to push it to. *If you get an error (or you did a typo like me) and trying to redo the step gives you ‘fatal: remote origin already exists.’, you can remove the origin and put in a new one again. Input ‘git remote rm origin’ and repeat step 4.
5. Input ‘git push -u origin master’ - this pushes your commit online, to Github. If successful, you can refresh your Github page online, you should be able to see your README.md file in the repository.

For VSC users: Tell Visual Studio Code which version you are using to code - click ‘View’, ‘Command Palette’, ‘Select Workspace Interpreter’, ‘Python 3.6’

*Optional for now (this tells git which files to ignore so that they are not uploaded to Github)
1. Go to ‘https://www.gitignore.io/'
2. Search what programs you use to code with - for me, it’s ‘Python’, ‘MacOS’, ‘VisualStudioCode’ - so they can customise the code for you.
3. Copy all the text generated
In Visual Studio Code:
4. Input ‘touch .gitignore’ to create the file
5. You can open the .gitignore file manually to paste it in (click ‘File’, ‘Open’, and open the entire folder. Do not click to open just the individual file. You want to the see the entire folder content on the left of the screen. Then double-click on .gitignore to see it.)
6. , or input ‘echo “text”>>.gitignore’ - this writes it into the file.
7. If there is a particular file in the directory you want to ignore (for e.g. testing.txt), type the filename in right at the very top ’testing.txt’

At this stage you have successfully set up your Github repository! You can now work locally on your computer, and push the code online to Github.

Next, we will host your work on the internet (via Heroku) so everyone can see it!

# Setting up heroku (in Python)

1. Signup for a free account/login
2. Click 'Create' on top right hand corner, and then 'Create new app'
3. Input app name; I kept the region as US. Click 'Create app’
4. Deployment method - click ’Connect to github’
5. Input repository name, press search. Click ‘Connect’.

   
In Visual Studio Code:
1. Input ‘pip3 install gunicorn’ to install gunicorn.
2. Create files required for Heroku. Input ‘touch requirements.txt’ - this is a file that is required for all Python apps. It details all the versions of libraries used for the app, which Heroku will load so that the app can run properly on the internet.
3. Input ‘pip3 freeze’ and search for the libraries you need (I.e. the stuff you import in your app), as well as gunicorn and flash. For example, my app needs to import pandas and numpy to run. I will copy and paste them into requirements.txt by inputing ’echo “numpy==1.13.1 pandas==0.20.3 gunicorn==19.7.1 Flask==0.12.2 Flask-Compress==1.4.0”>>requirements.txt
4. Input ‘touch runtime.txt’ - this is a file only required by Heroku. It tells Heroku which version of program to use.
       Input ‘python -V’ (that’s a capital V! I typo-ed and had to use exit()) to see what version you are on. Input ‘echo “python-3.6.2”>>runtime.txt’ to write it in the runtime.txt file. *If you are using Python 3 but the version on the computer shows up as 2, don’t worry - just type in the Python 3 version instead.
4. Input ‘touch app.py’ to create your app file. Input ‘echo “from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def main():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)”>>app.py. Otherwise, just copy and paste your py file into the folder if you already have your app written. *I tried to do this without Flask but didn’t manage to… if anyone has any advice as to how to do it, I would greatly appreciate your thoughts!

5. Input ‘mkdir templates’ - this creates a folder templates
6. Input ‘cd templates’ - to go into the templates folder. Input ‘touch index.html’ - creates an index.html file. Input ‘echo “<!DOCTYPE html>
<html>
<head>
    <title>My Homepage</title>
</head>
<body>
What’s up world!!!
</body>
</html>”>>index.html’
7. Input ‘cd ..’ To go back to your main folder.

8. Input ‘touch Procfile’ (that’s a capital P!). Input ‘echo “web: gunicorn app:app”>>Procfile’. Basically this tells Heroku this is a web app, using gunicorn (which reads Python), and the last bit is equivalent to ‘from app import app’ which is from app.py, import the variable app (defined as app = Flask(__name__). If you saved your app.py in another folder, you can replace the app.py bit with your folder location, like bot/app.py

9. Your app should be able to run locally now. Input ‘python3 app.py’ to try running it. It should give you the link, press command+click to open the link in a new page.
10. If everything works, go to the left side panel, there should be a blue bubble on a node icon detailing the number of changes you have made. Next to ‘Changes’, click the ‘+’ sign to stage all changes. Click the tick on top, input the message ‘added files for heroku deployment’. Press the three dots, click ‘Push to…’, click ‘origin’

In Heroku:
1. Press ‘Deploy Branch’ under Manual deploy. This is only required for the first time. *If you have troubles building it or running the application, click ‘More’ on the top right hand side, next to ‘Open app’, and click ‘View logs’. From there you can try to troubleshoot a little.
2. Once your app is successfully built, at the bottom it should provide the link to your website, saying deployed to Heroku. Go to that link to access your website! Alternatively, scroll up and click open app.
3. Once done, click ‘Enable automatic deploys’. This automatically updates your app every time you push to Github.
