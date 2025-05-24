from dotenv import load_dotenv
from pipeline.check_error_analyser import run_check_error_analyser_pipeline
from pipeline.check_question_analyser import run_check_question_analyser_pipeline
from pipeline.jupyter_data_pipeline import run_jupyter_data_pipeline


def main():
    load_dotenv()
    
    # run_jupyter_data_pipeline()
    # run_check_question_analyser_pipeline()
    run_check_error_analyser_pipeline()


if __name__ == "__main__":
    main()
