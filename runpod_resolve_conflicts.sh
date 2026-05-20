# Runpod Git Pull Resolution Script
# Run these commands on your Runpod server to resolve conflicts

# Step 1: Backup important local files (environment files)
echo "Backing up environment files..."
mkdir -p ~/backup_env
cp api/.env_alibaba ~/backup_env/ 2>/dev/null || true
cp api/web/.env_alibaba ~/backup_env/ 2>/dev/null || true

# Step 2: Stash local changes to tracked files
echo "Stashing local changes..."
git stash push -m "Local changes before pull" \
    api/USAGE.txt \
    api/api.py \
    api/module/__init__.py \
    api/module/ner/ner_system.py \
    api/module/ner/ner_train.py \
    api/module/ocr_system.py \
    api/requirements.txt \
    api/사용법.txt

# Step 3: Remove untracked files that exist in remote (but keep env files)
echo "Removing untracked files that conflict with remote..."
# Keep env files but remove others
rm -f api/.env_alibaba
rm -f api/web/.env_alibaba
rm -f api/__init__.py  # Will be pulled from remote
rm -f api/data/in/test.txt  # Should be ignored anyway
rm -rf api/model_downloaded/*  # Keep directory but remove contents
rm -rf api/module/ner/training/*  # Keep directory but remove contents
rm -rf api/module/ocr/*  # Will be pulled from remote
rm -f api/ner_test.py  # Will be pulled from remote
rm -f api/setup_env.py  # Will be pulled from remote
# Don't remove api/web/app.py and templates/index.html - these are modified locally

# Step 4: Now pull the changes
echo "Pulling latest changes..."
git pull origin main

# Step 5: Restore environment files if they were backed up
echo "Restoring environment files..."
if [ -f ~/backup_env/.env_alibaba ]; then
    cp ~/backup_env/.env_alibaba api/.env_alibaba
fi
if [ -f ~/backup_env/.env_alibaba ]; then
    cp ~/backup_env/.env_alibaba api/web/.env_alibaba
fi

# Step 6: Check if stash needs to be reapplied
echo "Checking stashed changes..."
git stash list

echo ""
echo "Done! Review the changes and if needed, run:"
echo "  git stash pop  # to reapply local changes"
echo "  git stash drop  # to discard stashed changes"

