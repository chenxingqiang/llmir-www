#!/bin/bash
set -e

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Deploying LLMIR website to GitHub Pages...${NC}"

# Save current git branch
CURRENT_BRANCH=$(git symbolic-ref --short HEAD)
echo -e "${GREEN}Current branch: $CURRENT_BRANCH${NC}"

# Build the website
echo -e "${BLUE}Building website from docs directory...${NC}"
cd docs
hugo --minify -d ../public
cd ..

# Add nojekyll file to disable Jekyll processing
echo -e "${BLUE}Adding .nojekyll file...${NC}"
touch public/.nojekyll

# Add CNAME file
echo -e "${BLUE}Adding CNAME file...${NC}"
echo "chenxingqiang.github.io" > public/CNAME

# Switch to gh-pages branch or create if it doesn't exist
echo -e "${BLUE}Switching to gh-pages branch...${NC}"
if git show-ref --quiet refs/heads/gh-pages; then
  git checkout gh-pages
  # Clean the branch, keeping only .git
  git rm -rf .
else
  git checkout --orphan gh-pages
  git rm -rf .
fi

# Copy public directory contents to root
echo -e "${BLUE}Copying built website files...${NC}"
cp -r public/* .
rm -rf public

# Add all files
echo -e "${BLUE}Committing changes...${NC}"
git add .
git commit -m "Deploy LLMIR website to GitHub Pages"

# Push to gh-pages
echo -e "${BLUE}Pushing to gh-pages branch...${NC}"
git push -f origin gh-pages

# Return to original branch
echo -e "${BLUE}Returning to $CURRENT_BRANCH branch...${NC}"
git checkout $CURRENT_BRANCH

echo -e "${GREEN}Deployment complete! Website should be live at: https://chenxingqiang.github.io/llmir-www/${NC}" 