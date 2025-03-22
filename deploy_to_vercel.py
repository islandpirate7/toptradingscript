#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Deployment script for S&P 500 Trading Strategy Web Interface
This script helps push the code to GitHub and deploy it to Vercel
"""

import os
import sys
import subprocess
import argparse
import json

def run_command(command, cwd=None):
    """Run a shell command and return the output"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def check_git_installed():
    """Check if Git is installed"""
    result = run_command("git --version")
    if not result:
        print("Git is not installed. Please install Git before continuing.")
        return False
    print(f"Git is installed: {result}")
    return True

def check_vercel_cli_installed():
    """Check if Vercel CLI is installed"""
    result = run_command("vercel --version")
    if not result:
        print("Vercel CLI is not installed. Would you like to install it? (y/n)")
        choice = input().lower()
        if choice == 'y':
            print("Installing Vercel CLI...")
            run_command("npm install -g vercel")
            return check_vercel_cli_installed()
        else:
            print("Vercel CLI is required for deployment. Please install it manually.")
            return False
    print(f"Vercel CLI is installed: {result}")
    return True

def setup_git_repo(repo_url, branch="main"):
    """Set up Git repository"""
    # Check if .git directory exists
    if os.path.exists(".git"):
        print("Git repository already initialized.")
    else:
        print("Initializing Git repository...")
        run_command("git init")
    
    # Check if remote origin exists
    remotes = run_command("git remote -v")
    if "origin" in remotes:
        print(f"Remote 'origin' already exists. Current remotes:\n{remotes}")
        print("Do you want to update the remote URL? (y/n)")
        choice = input().lower()
        if choice == 'y':
            run_command(f"git remote set-url origin {repo_url}")
            print(f"Updated remote 'origin' to {repo_url}")
    else:
        print(f"Adding remote 'origin' as {repo_url}")
        run_command(f"git remote add origin {repo_url}")
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists(".gitignore"):
        print("Creating .gitignore file...")
        with open(".gitignore", "w") as f:
            f.write("# Python\n")
            f.write("__pycache__/\n")
            f.write("*.py[cod]\n")
            f.write("*$py.class\n")
            f.write("*.so\n")
            f.write(".Python\n")
            f.write("env/\n")
            f.write("build/\n")
            f.write("develop-eggs/\n")
            f.write("dist/\n")
            f.write("downloads/\n")
            f.write("eggs/\n")
            f.write(".eggs/\n")
            f.write("lib/\n")
            f.write("lib64/\n")
            f.write("parts/\n")
            f.write("sdist/\n")
            f.write("var/\n")
            f.write("*.egg-info/\n")
            f.write(".installed.cfg\n")
            f.write("*.egg\n\n")
            f.write("# Logs\n")
            f.write("logs/\n")
            f.write("*.log\n\n")
            f.write("# Sensitive data\n")
            f.write("*credentials*.json\n")
            f.write("*api_key*.json\n\n")
            f.write("# Virtual Environment\n")
            f.write("venv/\n")
            f.write("ENV/\n\n")
            f.write("# IDE files\n")
            f.write(".idea/\n")
            f.write(".vscode/\n")
            f.write("*.swp\n")
            f.write("*.swo\n")
    
    return True

def commit_and_push(commit_message, branch="main"):
    """Commit and push changes to GitHub"""
    print("Adding all files to Git...")
    run_command("git add .")
    
    print(f"Committing with message: '{commit_message}'")
    run_command(f'git commit -m "{commit_message}"')
    
    print(f"Pushing to branch '{branch}'...")
    result = run_command(f"git push -u origin {branch}")
    
    if result:
        print("Successfully pushed to GitHub!")
        return True
    else:
        print("Failed to push to GitHub. Please check your repository settings and try again.")
        return False

def setup_vercel_config():
    """Set up Vercel configuration"""
    if not os.path.exists("vercel.json"):
        print("Creating vercel.json configuration file...")
        config = {
            "version": 2,
            "builds": [
                {
                    "src": "web_interface/app.py",
                    "use": "@vercel/python"
                }
            ],
            "routes": [
                {
                    "src": "/(.*)",
                    "dest": "web_interface/app.py"
                }
            ],
            "env": {
                "FLASK_ENV": "production"
            }
        }
        
        with open("vercel.json", "w") as f:
            json.dump(config, f, indent=2)
        
        print("Created vercel.json configuration file.")
    else:
        print("vercel.json already exists.")
    
    return True

def deploy_to_vercel():
    """Deploy to Vercel"""
    print("Deploying to Vercel...")
    print("This will open a browser window for authentication if needed.")
    
    # First, log in to Vercel if needed
    run_command("vercel login")
    
    # Deploy the project
    result = run_command("vercel --prod")
    
    if result:
        print("Successfully deployed to Vercel!")
        print("Your application is now live at the URL provided above.")
        return True
    else:
        print("Failed to deploy to Vercel. Please check the error messages and try again.")
        return False

def main():
    parser = argparse.ArgumentParser(description='Deploy S&P 500 Trading Strategy Web Interface to Vercel')
    parser.add_argument('--repo-url', required=True, help='GitHub repository URL (e.g., https://github.com/username/repo.git)')
    parser.add_argument('--branch', default='main', help='Git branch to push to (default: main)')
    parser.add_argument('--commit-message', default='Deploy to Vercel', help='Git commit message')
    parser.add_argument('--skip-git', action='store_true', help='Skip Git setup and push')
    parser.add_argument('--skip-vercel', action='store_true', help='Skip Vercel deployment')
    args = parser.parse_args()
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("=== S&P 500 Trading Strategy Web Interface Deployment ===")
    
    if not args.skip_git:
        # Check if Git is installed
        if not check_git_installed():
            return
        
        # Set up Git repository
        if not setup_git_repo(args.repo_url, args.branch):
            return
        
        # Commit and push changes
        if not commit_and_push(args.commit_message, args.branch):
            return
    
    if not args.skip_vercel:
        # Check if Vercel CLI is installed
        if not check_vercel_cli_installed():
            return
        
        # Set up Vercel configuration
        if not setup_vercel_config():
            return
        
        # Deploy to Vercel
        if not deploy_to_vercel():
            return
    
    print("=== Deployment process completed! ===")

if __name__ == "__main__":
    main()
