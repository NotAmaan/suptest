# GitHub Actions for SUPIR-Demo

This directory contains GitHub Actions workflows for building and deploying the SUPIR-Demo Docker image to GitHub Container Registry (ghcr.io).

## Workflows

### 1. docker-build.yml
Basic workflow that builds and pushes the Docker image on:
- Push to main branch
- Pull requests (build only, no push)
- Git tags (v*)

### 2. docker-build-runpod.yml
Enhanced workflow specifically for RunPod deployment with:
- Manual trigger option with custom tags
- RunPod-specific image tagging
- Deployment instructions in workflow summary
- Optional push to registry

## Setup Instructions

### 1. Enable GitHub Container Registry
No additional secrets needed! The workflows use the built-in `GITHUB_TOKEN` for authentication.

### 2. Set Package Visibility (Optional)
After your first image push, you may want to:
1. Go to your GitHub profile > Packages
2. Find your package
3. Package settings > Change visibility (public/private)

### 3. Enable GitHub Actions
Ensure GitHub Actions is enabled in your repository settings.

## Usage

### Automatic Builds
The workflow automatically runs when:
- You push to the main branch
- You create a version tag (e.g., `v1.0.0`)

### Manual Builds
To manually trigger a build:
1. Go to Actions tab in your repository
2. Select "Build RunPod Docker Image"
3. Click "Run workflow"
4. Optionally specify:
   - Whether to push to registry
   - Custom image tag

### Image Tags
The workflow creates the following tags:
- `latest` - Updated on each push to main
- `runpod-latest` - RunPod-specific latest tag
- `main` - Tracks the main branch
- `v1.0.0` - Semantic version tags
- `1.0` - Major.minor version tags
- Custom tags when manually triggered

## Deployment on RunPod

After the image is built and pushed:

### For Pod Mode (Interactive):
```bash
# Image format: ghcr.io/GITHUB_USERNAME/REPO_NAME:TAG
docker pull ghcr.io/your-github-username/supir-demo:latest
```
- Environment: `MODE_TO_RUN=pod`
- Expose port: 7860
- Mount volume at: `/workspace`

### For Serverless Mode:
```bash
docker pull ghcr.io/your-github-username/supir-demo:latest
```
- Environment: `MODE_TO_RUN=serverless`
- Mount volume at: `/workspace`

### Authentication for Private Images
If your image is private, you'll need to authenticate:
```bash
# On RunPod or locally
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

## Troubleshooting

1. **Build fails with permissions error**
   - Ensure your repository has Actions permissions enabled
   - Check Settings > Actions > General > Workflow permissions
   - Select "Read and write permissions"

2. **Out of space errors**
   - The image is large (~15GB). GitHub Actions provides ~14GB of space
   - Consider using self-hosted runners for larger builds
   - Use multi-stage builds to reduce final image size

3. **Slow builds**
   - The workflow uses GitHub Actions cache to speed up builds
   - First builds will be slower; subsequent builds use cache

4. **Package not visible**
   - New packages are private by default
   - Change visibility in your GitHub profile > Packages > Package settings