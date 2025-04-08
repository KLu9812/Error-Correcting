from error_correct import *
from generate_data import *



if __name__ == "__main__":
    models = ["gemma-2-27b-it", "qwq-32b", "qwen2.5-coder-32b-instruct", "light-r1-32b"]
    
    
    #Cladder run--------------------------------------------------------------
    # save_cladder(100)
    
    # cladder_info = load_cladder_info()
    # ec = ErrorCorrect(models = models, schema = yes_no_response_schema,
    #                   questions = cladder_info['questions'], correct_answers = cladder_info["correct answers"])
    
    # ec.set_initial_response_prompt(yes_no_role_message)
    # ec.save_responses()
    
    # ec.set_multi_input_role(yes_no_multi_input_role_message)
    # ec.set_multi_input_prompt(yes_no_multi_input_prompt)
    # ec.get_multi_input()
    
    # ec.analyze()
    
    #AIME run-----------------------------------------------------------------
    save_aime(100)
    
    aime_info = load_aime_info()
    ec = ErrorCorrect(models = models, schema = yes_no_response_schema,
                      questions = aime_info['questions'], correct_answers = aime_info["correct answers"])
    
    ec.set_initial_response_prompt(digit_role_message)
    ec.save_responses()
    
    ec.set_multi_input_role(digit_multi_input_role_message)
    ec.set_multi_input_prompt(digit_multi_input_prompt)
    ec.get_multi_input()
    
    ec.analyze()