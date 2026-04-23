from agents.challenger import run_challenger
from  config.qwen_config import test_model, download_backbone

def main():
    download_backbone()
    #response = test_model(prompt="Get me the Weather in Freiberg, Sachs, Germany")

    response = run_challenger()
    print(response)

if __name__ == "__main__":
    main()