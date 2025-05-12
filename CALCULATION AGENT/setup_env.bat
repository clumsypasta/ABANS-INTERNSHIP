@echo off
echo Setting up environment for Trade Position Calculator...

if exist .env (
    echo .env file already exists.
    echo If you want to update it, please edit it manually or delete it first.
    goto :end
)

echo Creating .env file...
echo Please enter your Google API key for Gemini:
set /p API_KEY="> "

if "%API_KEY%"=="" (
    echo No API key provided. Using placeholder value.
    echo You'll need to update the .env file manually before using the AI Assistant.
    set API_KEY=your_api_key_here
)

echo # Environment variables for Trade Position Calculator > .env
echo. >> .env
echo # Google API key for Gemini AI >> .env
echo GOOGLE_API_KEY=%API_KEY% >> .env

echo .env file created successfully!
echo You can now run the application with: run.bat

:end
pause
