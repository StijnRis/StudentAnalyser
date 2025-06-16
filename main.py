from dotenv import load_dotenv

from pipeline.jupyter_data_pipeline import run_jupyter_data_pipeline


def main():
    load_dotenv()

    run_jupyter_data_pipeline()


if __name__ == "__main__":
    main()
