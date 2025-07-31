from data.parser_test import parse_and_build, MissingSlot  # refactored for pure parsing!
from llm_main import LLMRouter
import functions


class Dispatcher:
    def __init__(self, classifier, llm_router):
        self.classifier = classifier
        self.llm_router = llm_router

    def classify(self, raw_query):
        return self.classifier.classify(raw_query)

    def pure_parse(self, raw_query, func_name):
        # Refactor parse_and_build to not call input(), but raise MissingSlot
        # For now, pseudo-code:
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
            'Asaoka_data': functions.asaoka_data,
            'reporter_Asaoka': functions.reporter_asaoka,
            'plot_combi_S': functions.plot_combi_S
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