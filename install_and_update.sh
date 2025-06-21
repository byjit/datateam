#!/bin/bash

# Check if a package name is provided as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <package_name>"
  exit 1
fi

PACKAGE_NAME="$1"

echo "Installing $PACKAGE_NAME using uv pip..."
uv pip install "$PACKAGE_NAME"

if [ $? -eq 0 ]; then
  echo "$PACKAGE_NAME installed successfully."
  echo "Updating requirements.txt..."
  uv pip freeze > requirements.txt
  echo "requirements.txt updated."
else
  echo "Failed to install $PACKAGE_NAME."
  exit 1
fi
