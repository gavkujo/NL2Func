from data.parser_test import parse_and_build, MissingSlot  # refactored for pure parsing!
from llm_main import LLMRouter
import functions
from main import choose_function, rule_based_func


class Dispatcher:
    def __init__(self, classifier, llm_router):
        self.classifier = classifier
        self.llm_router = llm_router

    def classify(self, raw_query):
        result = choose_function(raw_query, self.classifier)
        if isinstance(result, tuple):
            classifier_func, rule_func = result
            print(f"System detected a possible function clash.")
            print(f"1. {classifier_func} (Classifier)")
            print(f"2. {rule_func} (Rule-based)")
            choice = input("Choose which function to run (type 1 or 2): ").strip()
            if choice == "1":
                return classifier_func, classifier_func
            elif choice == "2":
                return rule_func, rule_func
            else:
                print("Invalid choice, defaulting to classifier output.")
                return classifier_func, classifier_func
        return result, result

    def pure_parse(self, raw_query, func_name):
        try:
            return parse_and_build(raw_query, func_name)
        except ValueError as e:
            # parse_and_build should raise MissingSlot(slot) instead of ValueError
            raise

    def gather_params(self, raw_query, func_name):
        aux_ctx = raw_query
        collected = {}
        while True:
            try:
                params = self.pure_parse(aux_ctx, func_name)
                return params
            except MissingSlot as ms:
                answer = input(f"What’s your {ms.slot}? ")
                if answer.lower() in ["skip", "never mind"]:
                    return None
                collected[ms.slot] = answer
                aux_ctx += f"\n{ms.slot}: {answer}"

    def run_function(self, func_name, params):
        # Map function name to actual function in functions.py
        func_map = {
            'Asaoka_data': functions.Func1,
            'reporter_Asaoka': functions.Func2,
            'plot_combi_S': functions.Func3,
            'SM_overview': functions.Func4
        }
        func = func_map.get(func_name)
        if func and params:
            return func(**params)
        return None

    def build_and_send(self, raw_query, func_name, params, func_output):
        classifier_data = {"Function": func_name, "Params": params, "Output": func_output}
        self.llm_router.handle_user(
            raw_query,
            func_name=func_name,
            classifier_data=classifier_data,
            func_output=func_output
        )