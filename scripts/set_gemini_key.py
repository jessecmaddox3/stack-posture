#!/usr/bin/env python3
"""Explicit local key setup; never called by normal setup, tests or the demo."""
from getpass import getpass
from posture.secrets import store_api_key


def main():
    print('Store your own Gemini API key in macOS Keychain. This does not enable uploads.')
    key = getpass('API key (hidden; Return cancels): ').strip()
    if key:
        store_api_key(key)
        print('Stored. Enable gemini_enabled separately only if you want image uploads.')
    else:
        print('Canceled.')


if __name__ == '__main__':
    main()
