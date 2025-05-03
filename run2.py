from error_correct import *
from generate_data import *



if __name__ == "__main__":
    models = ["./QwQ-32B"]
    
    
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
    
    # ec.clean_yes_no()
    # ec.analyze()
    
    #AIME run-----------------------------------------------------------------
    #save_aime(5)
    
    aime_info = load_aime_info()
    ec = ErrorCorrect(models = models, schema = yes_no_response_schema,
                      questions = aime_info['questions'], correct_answers = aime_info["correct answers"])
    ec.set_current_model("./QwQ-32B")

    ec.set_initial_response_role(digit_role_message)
    #ec.save_responses()
    #ec.create_batch_prompts()
    ec.save_batch_responses()
    
    #ec.set_multi_input_role(digit_multi_input_role_message)
    #ec.set_multi_input_prompt(digit_multi_input_prompt)
    #ec.get_multi_input()
    
    #ec.analyze()
