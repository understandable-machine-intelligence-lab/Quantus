#!/usr/bin/env zsh
set -e
# Check no un-committed changes
echo -n "Checking if there are uncommited changes... "
trap 'echo -e "\033[0;31mCHANGED\033[0m"' ERR
git diff-index --quiet HEAD --
trap - ERR
echo -e "\033[0;32mUNCHANGED\033[0m"
# Check provided 1 positional argument
if [ $# -eq 0 ]; then
  echo -e "Must provide tag as positional argument"
fi
if ! command -v gh &> /dev/null
then
    echo "GitHub CLI not installed."
    exit
fi
TAG=$1
echo "TAG=${TAG}"
# Update main ref's and switch to main's HEAD.
git fetch --atomic --verbose && git checkout main
# Clean old artifacts.
rm -f -R build
# Build wheel.
python3 -m pip install tox
python3 -m tox run -e build
# Tag release.
git tag "$TAG"
git push --follow-tags
# Create GitHub release draft.
gh release create "$TAG" ./dist/* --draft --latest
