import os
from dotenv import load_dotenv, find_dotenv


def setup_api_keys():
    # Find .env file
    env_path = find_dotenv()

    if not env_path:
        print("Creating new .env file...")
        env_path = ".env"
        with open(env_path, "w") as f:
            f.write("# LLM API Keys\n")

    print("\nLLM API Key Setup")
    print("=" * 50)

    keys = {}
    keys['OPENAI_API_KEY'] = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    keys['ANTHROPIC_API_KEY'] = input("Enter your Anthropic API key (or press Enter to skip): ").strip()
    keys['GOOGLE_API_KEY'] = input("Enter your Google API key (or press Enter to skip): ").strip()
    keys['GROK_API_KEY'] = input("Enter your Grok API key (or press Enter to skip): ").strip()

    # Read existing .env file
    env_lines = []
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_lines = f.readlines()

    # Update or add keys
    updated_keys = set()
    for i, line in enumerate(env_lines):
        for key_name, key_value in keys.items():
            if line.strip().startswith(f"{key_name}=") and key_value:
                env_lines[i] = f"{key_name}={key_value}\n"
                updated_keys.add(key_name)

    # Add new keys that weren't updated
    for key_name, key_value in keys.items():
        if key_name not in updated_keys and key_value:
            env_lines.append(f"{key_name}={key_value}\n")

    # Write back
    with open(env_path, "w") as f:
        f.writelines(env_lines)

    print(f"\nAPI keys saved to {env_path}")
    print("Set USE_MOCK_MODE=false in .env to use real API calls")
    print("Set USE_MOCK_MODE=true to use mock responses (when testing)")


if __name__ == "__main__":
    setup_api_keys()