#!/usr/bin/env zsh
set -e
# Check no un-committed changes
echo -n "Checking if there are uncommited changes... "
trap 'echo -e "\033[0;31mCHANGED\033[0m"' ERR
git diff-index --quiet HEAD --
trap - ERR
echo -e "\033[0;32mUNCHANGED\033[0m"

echo -n "Looking for GitHub CLI... "
if ! command -v gh &>/dev/null; then
  echo -e "\033[0;31m GitHub CLI not installed.\033[0m"
  exit
else
  echo -e "\033[0;32OK\033[0m"
fi
# Check provided 1 positional argument
if [ $# -eq 0 ]; then
  echo -e "\033[0;31m Must provide tag as positional argument\033[0m"
  exit
fi
TAG=$1
echo -n "TAG=${TAG}"
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
