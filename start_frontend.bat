@echo off
cd /d "%~dp0frontend"
if not exist node_modules call npm install
npm run dev
