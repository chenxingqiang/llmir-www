# LLMIR WWW

This contains the source code for https://chenxingqiang.github.io/llmir-www/ which is rendered from the `gh-pages` branch of the same repo using GitHub Pages.

To contribute, feel free to fork this repository and send a pull-request.

The website is deployed when changes are pushed to the `gh-pages` branch. You can manually trigger a build and deployment by following the instructions below.

We are using the [Hugo](https://gohugo.io/) framework for generating the website. The source pages are written in Markdown format under the `docs/content` folder.

## Local Development

To preview the website locally:

1. Install [Hugo](https://gohugo.io/getting-started/installing/) on your machine
2. Navigate to the `docs` directory:
   ```sh
   cd docs
   ```
3. Run the Hugo server:
   ```sh
   hugo server
   ```
4. Access the local version of the website at http://localhost:1313/

Any changes you make to the source Markdown will automatically be refreshed by the local Hugo server.

## Deployment

### Automatic Deployment Script

We provide a shell script to automate the deployment process:

```sh
./gh-pages.sh
```

This script will build the website, create a clean `gh-pages` branch, and push the built content to GitHub.

### Manual Deployment

If you prefer to deploy manually:

1. Make your changes to the Markdown files in the `docs/content` directory
2. Build the website:
   ```sh
   cd docs
   hugo --minify -d ../public
   ```
3. Add a `.nojekyll` file to disable GitHub Pages Jekyll processing:
   ```sh
   touch ../public/.nojekyll
   ```
4. Push the contents of the `public` directory to the `gh-pages` branch

Alternatively, you can use the GitHub Actions workflow defined in `.github/workflows/main.yml` to automate this process.
